using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using PimDeWitte.UnityMainThreadDispatcher;
using UnityEngine;

/// <summary>
/// StreamDiffusion客户端，用于与Python服务端通信并控制图像生成
/// 所有参数都可在Unity Inspector中配置：
/// - 模型路径：相对于StreamingAssets/models目录的路径
/// - 图像尺寸：生成图像的宽高
/// - 种子：随机种子，影响生成效果
/// - 加速模式：使用的加速技术，如tensorrt, cuda等
/// - 提示词：控制图像生成内容的文本描述
/// - 强度参数：控制生成过程的各种强度系数
/// </summary>
public class StreamDiffusionClient : MonoBehaviour
{
    [Tooltip("基础模型路径，相对于StreamingAssets/models目录")]
    public string _baseModelPath = "Model/photonLCM_v10.safetensors";

    [Tooltip("VAE模型路径，相对于StreamingAssets/models目录")]
    public string _tinyVaeModelPath = "VAE";

    [Tooltip("LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath = "LoRA/tbh123-.safetensors";

    [Tooltip("加速模式：tensorrt, cuda等")]
    public string _acceleration = "tensorrt";

    [Tooltip("图像宽度")]
    public int _width = 512;

    [Tooltip("图像高度")]
    public int _height = 512;

    [Tooltip("随机种子")]
    public int _seed = 603665;

    [Tooltip("是否使用TinyVAE")]
    public bool _useTinyVae = true;

    [Tooltip("是否使用LCM LoRA")]
    public bool _useLcmLora = true;

    [Tooltip("LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale = 0.85f;

    [Tooltip("是否显示Python控制台")]
    public bool _showPythonConsole = false;

    [Tooltip("图像生成强度")]
    [Range(1f, 3.0f)]
    public float _strength = 1.0f;

    [Tooltip("提示词")]
    public string _defaultPrompt = "";

    [Tooltip("负面提示词")]
    public string _defaultNegativePrompt = "";

    public Material _resultMaterial;
    private Texture2D _resultTexture;
    private System.Diagnostics.Process _backgroundProcess;

    private string _serverIP = "127.0.0.1";
    private int _serverPort = 9999;
    private int _pipelineLoaded = 0;
    private bool _isRunning = false,
        _isAdvancing = false;
    private bool _restartRequested = false; // 标记是否请求重启服务

    // 添加Socket配置参数
    private int _receiveBufferSize = 65536; // 增大接收缓冲区
    private int _receiveTimeout = 120000; // 接收超时时间(毫秒)
    private int _sendTimeout = 30000; // 发送超时时间(毫秒)
    private int _maxRetryCount = 3; // 最大重试次数

    private TcpClient _client;
    private NetworkStream _stream;
    private Thread _clientThread;
    private byte[] _advancedData;

    public bool isValid()
    {
        return _isRunning;
    }

    public bool isGenerating()
    {
        return _isAdvancing;
    }

    public bool isPending()
    {
        return _isRunning && _pipelineLoaded < 0;
    }

    public bool isRunning()
    {
        return _isRunning && _pipelineLoaded > 0;
    }

    // 辅助方法：构建完整的模型路径
    private string GetFullModelPath(string relativePath)
    {
        return System
            .IO.Path.Combine(Application.streamingAssetsPath, "models", relativePath)
            .Replace("\\", "/");
    }

    private void ConnectToServer()
    {
        try
        {
            _client.Connect(_serverIP, _serverPort);
            _stream = _client.GetStream();
            _isRunning = true;

            // 发送模型路径
            setModelPaths();

            Thread receiveThread = new Thread(new ThreadStart(ReceiveMessages));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error connecting to server: " + e.Message);
        }
    }

    private void ReceiveMessages()
    {
        try
        {
            byte[] tempBuffer = new byte[_receiveBufferSize];

            while (_isRunning)
            {
                if (_stream.DataAvailable)
                {
                    try
                    {
                        int bytesRead = _stream.Read(tempBuffer, 0, tempBuffer.Length);
                        if (bytesRead > 0)
                        {
                            byte[] receivedBytes = new byte[bytesRead];
                            System.Buffer.BlockCopy(tempBuffer, 0, receivedBytes, 0, bytesRead);

                            UnityMainThreadDispatcher
                                .Instance()
                                .Enqueue(() =>
                                {
                                    UpdateServerData(receivedBytes, bytesRead);
                                });
                        }
                    }
                    catch (System.IO.IOException ioEx)
                    {
                        Debug.LogWarning($"IO异常: {ioEx.Message}，尝试继续接收...");
                        Thread.Sleep(100); // 短暂暂停后继续
                    }
                }
                else
                {
                    // 避免CPU占用过高
                    Thread.Sleep(10);
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"接收消息时出错: {e.Message}");

            // 如果不是手动关闭连接导致的异常，尝试重新连接
            if (_isRunning && !_restartRequested)
            {
                Debug.Log("尝试重新连接服务器...");
                TryReconnect();
            }
        }
    }

    private void UpdateServerData(byte[] data, int length)
    {
        byte[] buffer = new byte[length];
        System.Buffer.BlockCopy(data, 0, buffer, 0, length);
        if (_pipelineLoaded <= 0)
        {
            if (length < 10)
            {
                string result = Encoding.UTF8.GetString(buffer);
                if (result == "loaded")
                    _pipelineLoaded = 1;
                else
                    _pipelineLoaded = 0;
            }
        }
        else if (length > 10)
        {
            byte[] lastFourBytes = new byte[4];
            System.Array.Copy(buffer, buffer.Length - 4, lastFourBytes, 0, 4);
            bool validEnd = true;
            byte pipeByte = (byte)'|';
            for (int i = 0; i < 4; i++)
            {
                if (lastFourBytes[i] != pipeByte)
                    validEnd = false;
            }

            int len0 = (_advancedData == null ? 0 : _advancedData.Length);
            int len1 = (validEnd ? (buffer.Length - 4) : buffer.Length);
            byte[] totalData = new byte[len0 + len1];
            if (_advancedData != null)
                System.Buffer.BlockCopy(_advancedData, 0, totalData, 0, len0);
            System.Buffer.BlockCopy(buffer, 0, totalData, len0, len1);
            _advancedData = totalData;

            if (validEnd)
            {
                if (_resultTexture == null)
                {
                    _resultTexture = new Texture2D(2, 2);
                    if (_resultMaterial != null)
                        _resultMaterial.mainTexture = _resultTexture;
                }

                if (_resultTexture.LoadImage(_advancedData))
                    _resultTexture.Apply();
                _advancedData = null;
                _isAdvancing = false;
                //Debug.Log("Received image: " + len0);
            }
        }
        else
        {
            string result = Encoding.UTF8.GetString(buffer);
            _advancedData = null;
            _isAdvancing = false;
            Debug.Log("Received message: " + result);
        }
    }

    public void setModelPaths()
    {
        // 构建完整路径
        string fullBaseModelPath = GetFullModelPath(_baseModelPath);
        string fullVaePath = GetFullModelPath(_tinyVaeModelPath);
        string fullLoraPath = GetFullModelPath(_loraModelPath);

        Debug.Log(
            $"Sending paths to Python server: Base={fullBaseModelPath}, VAE={fullVaePath}, LoRA={fullLoraPath}"
        );

        string cmd0 =
            $"|start|command||paths||base_model||{fullBaseModelPath}||taesd_model||{fullVaePath}"
            + $"||lora_model||{fullLoraPath}||run||0|end|";
        SendCommandToPython(cmd0);
    }

    public void LoadPipeline()
    {
        int vae = (_useTinyVae ? 1 : 0),
            lora = (_useLcmLora ? 1 : 0);

        Debug.Log(
            $"准备发送参数到Python: 宽度={_width}, 高度={_height}, 种子={_seed}, 强度={_strength}, LoRA强度={_loraScale}"
        );

        // 确保种子值作为字符串发送
        string seedStr = _seed.ToString();

        string loadCmd =
            $"|start|command||load||width||{_width}||height||{_height}||seed||{seedStr}"
            + $"||use_vae||{vae}||use_lora||{lora}||strength||{_strength}||lora_scale||{_loraScale}||acceleration||{_acceleration}"
            + $"||prompt||{_defaultPrompt}||neg_prompt||{_defaultNegativePrompt}||run||0|end|";

        Debug.Log($"发送命令长度: {loadCmd.Length} 字节");
        _pipelineLoaded = -1; // pending
        SendCommandToPython(loadCmd);
    }

    // 添加新方法以安全地发送命令
    private void SendCommandToPython(string command)
    {
        try
        {
            if (_stream != null && _stream.CanWrite)
            {
                // 使用UTF-8编码，确保特殊字符和中文能正确传输
                byte[] messageBytes = Encoding.UTF8.GetBytes(command);
                _stream.Write(messageBytes, 0, messageBytes.Length);
                Debug.Log($"Command sent, length: {messageBytes.Length} bytes");
            }
            else
            {
                Debug.LogError("Cannot send command - stream is null or not writable");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending command to Python: {e.Message}");
        }
    }

    // 扩展AdvancePipeline方法，支持动态传递强度参数
    public void AdvancePipeline(Texture2D tex, string prompt, float? strength = null, float? loraScale = null)
    {
        try
        {
            if (_stream == null || !_stream.CanWrite || tex == null)
            {
                Debug.LogError(
                    "Cannot advance pipeline - stream is null or not writable, or texture is null"
                );
                return;
            }

            // 使用传入的参数或默认值
            float currentStrength = strength ?? _strength;
            float currentLoraScale = loraScale ?? _loraScale;
            
            Debug.Log($"动态生成: 提示词='{prompt}', 强度={currentStrength}, LoRA强度={currentLoraScale}");

            // 将图像编码为JPEG，降低质量以减少数据大小
            byte[] imageBytes = ImageConversion.EncodeToJPG(tex, 90); // 90%质量，减少数据量
            if (imageBytes == null || imageBytes.Length == 0)
            {
                Debug.LogError("Failed to encode image to JPEG");
                return;
            }

            // 使用Base64编码处理图像数据，避免二进制数据中的特殊字符干扰协议解析
            string base64Image = System.Convert.ToBase64String(imageBytes);
            Debug.Log(
                $"Encoded image size: {imageBytes.Length} bytes, Base64 length: {base64Image.Length}"
            );

            // 构建命令，使用Base64编码的图像数据，添加强度参数
            string command =
                $"|start|command||advance||prompt||{prompt}||strength||{currentStrength}||lora_scale||{currentLoraScale}||image_base64||{base64Image}||run||0|end|";

            // 分块发送大数据
            byte[] commandBytes = Encoding.UTF8.GetBytes(command);
            int chunkSize = 16384; // 16KB一个块
            int sentBytes = 0;

            while (sentBytes < commandBytes.Length)
            {
                int remaining = commandBytes.Length - sentBytes;
                int currentChunkSize = System.Math.Min(remaining, chunkSize);

                _stream.Write(commandBytes, sentBytes, currentChunkSize);
                sentBytes += currentChunkSize;

                // 每发送一个块后稍微等待以防止缓冲区溢出
                if (sentBytes < commandBytes.Length)
                {
                    Thread.Sleep(5);
                }
            }

            Debug.Log($"Sent advance request in chunks, total size: {commandBytes.Length} bytes");

            _isAdvancing = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error in AdvancePipeline: {e.Message}");
            _isAdvancing = false;
        }
    }

    private void OutputDataReceived(object sender, System.Diagnostics.DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data))
            Debug.Log(e.Data);
    }

    private void StartTcpClient()
    {
        Debug.Log("Start TCP client...");
        _client = new TcpClient();

        // 配置Socket参数
        _client.ReceiveBufferSize = _receiveBufferSize;
        _client.SendBufferSize = _receiveBufferSize;
        _client.ReceiveTimeout = _receiveTimeout;
        _client.SendTimeout = _sendTimeout;
        _client.NoDelay = true; // 禁用Nagle算法，提高小数据包发送效率

        _clientThread = new Thread(new ThreadStart(ConnectToServer));
        _clientThread.IsBackground = true;
        _clientThread.Start();
    }

    void Update()
    {
        // 检测键盘快捷键 - Ctrl+R 重启服务
        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKeyDown(KeyCode.R))
        {
            Debug.Log("重启StreamDiffusion服务的快捷键被按下 (Ctrl+R)");
            RestartStreamDiffusionService();
        }
    }

    // 重启StreamDiffusion服务的方法
    public void RestartStreamDiffusionService()
    {
        Debug.Log("正在重启StreamDiffusion服务...");
        _restartRequested = true;

        // 关闭现有连接
        _isRunning = false;
        _pipelineLoaded = 0;

        if (_client != null && _client.Connected)
        {
            try
            {
                if (_stream != null)
                    _stream.Close();
                _client.Close();
            }
            catch (System.Exception e)
            {
                Debug.LogError($"关闭连接时出错: {e.Message}");
            }
        }

        // 重启Python进程
        if (_backgroundProcess != null && !_backgroundProcess.HasExited)
        {
            try
            {
                _backgroundProcess.Kill();
                _backgroundProcess.WaitForExit(5000); // 等待最多5秒
            }
            catch (System.Exception e)
            {
                Debug.LogError($"终止Python进程时出错: {e.Message}");
            }
        }

        // 启动新的Python进程
        StartPythonProcess();

        // 延迟启动TCP客户端
        _advancedData = null;
        _isAdvancing = false;
        Invoke("StartTcpClient", 8.0f);

        _restartRequested = false;
        Debug.Log("StreamDiffusion服务重启请求已处理");
    }

    private void StartPythonProcess()
    {
        var pythonHome = $"{Application.streamingAssetsPath}/envs/streamdiffusion";
        var exePath = $"{pythonHome}/python.exe";

        if (System.IO.File.Exists(exePath))
        {
            Debug.Log("启动背景服务器...");
            System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo(
                exePath
            );
            startInfo.FileName = exePath;
            startInfo.Arguments = "image_predictor.py";
            startInfo.WorkingDirectory = $"{Application.streamingAssetsPath}";

            if (_showPythonConsole)
            {
                startInfo.UseShellExecute = true;
                startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
            }
            else
            {
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                startInfo.RedirectStandardOutput = true;
                startInfo.RedirectStandardError = true;
                startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
            }

            _backgroundProcess = new System.Diagnostics.Process();
            _backgroundProcess.StartInfo = startInfo;

            if (!_showPythonConsole)
            {
                _backgroundProcess.OutputDataReceived += OutputDataReceived;
                _backgroundProcess.ErrorDataReceived += OutputDataReceived;
            }

            if (!_backgroundProcess.Start())
            {
                Debug.LogError("启动image_predictor.py失败");
            }

            if (!_showPythonConsole)
            {
                _backgroundProcess.BeginOutputReadLine();
                _backgroundProcess.BeginErrorReadLine();
            }
        }
        else
        {
            Debug.LogError("找不到Python.exe");
        }
    }

    // 添加重连机制
    private void TryReconnect()
    {
        int retryCount = 0;
        bool reconnected = false;

        while (!reconnected && retryCount < _maxRetryCount && _isRunning && !_restartRequested)
        {
            try
            {
                Debug.Log($"重新连接尝试 #{retryCount + 1}...");
                if (_client != null)
                {
                    _client.Close();
                }

                _client = new TcpClient();
                _client.ReceiveBufferSize = _receiveBufferSize;
                _client.SendBufferSize = _receiveBufferSize;
                _client.ReceiveTimeout = _receiveTimeout;
                _client.SendTimeout = _sendTimeout;
                _client.NoDelay = true;

                _client.Connect(_serverIP, _serverPort);
                _stream = _client.GetStream();

                reconnected = true;
                Debug.Log("重新连接成功！");

                // 重新启动接收线程
                Thread receiveThread = new Thread(new ThreadStart(ReceiveMessages));
                receiveThread.IsBackground = true;
                receiveThread.Start();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"重连尝试 #{retryCount + 1} 失败: {ex.Message}");
                retryCount++;
                Thread.Sleep(2000); // 等待2秒后重试
            }
        }

        if (!reconnected && _isRunning && !_restartRequested)
        {
            Debug.LogError("重连失败，将重启服务...");

            // 从主线程调用重启
            UnityMainThreadDispatcher
                .Instance()
                .Enqueue(() =>
                {
                    RestartStreamDiffusionService();
                });
        }
    }

    void Start()
    {
        // 设置应用程序在后台继续运行，避免失去焦点时造成的超时问题
        Application.runInBackground = true;

        var pythonHome = $"{Application.streamingAssetsPath}/envs/streamdiffusion";
        var projectHome = $"{Application.streamingAssetsPath}/streamdiffusion";
        var scripts = $"{pythonHome}/Scripts";

        var path = System.Environment.GetEnvironmentVariable("PATH")?.TrimEnd(';');
        path = string.IsNullOrEmpty(path)
            ? $"{pythonHome};{scripts}"
            : $"{pythonHome};{scripts};{path}";
        System.Environment.SetEnvironmentVariable(
            "PATH",
            path,
            System.EnvironmentVariableTarget.Process
        );

        StartPythonProcess();
        Invoke("StartTcpClient", 8.0f);
    }

    void OnDestroy()
    {
        _isRunning = false;
        _pipelineLoaded = 0;
        if (_backgroundProcess != null && !_backgroundProcess.HasExited)
            _backgroundProcess.Kill();
        if (_clientThread != null)
            _clientThread.Interrupt();
        if (_client != null)
            _client.Close();
    }
}
