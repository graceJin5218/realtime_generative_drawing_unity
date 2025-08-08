/*using System.Collections;
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
    public string _baseModelPath = "kohaku-v2.1";

    [Tooltip("VAE模型路径，相对于StreamingAssets/models目录")]
    public string _tinyVaeModelPath = "taesd";

    [Tooltip("第一个LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath = "lcm-lora-sdv1-5";

    [Tooltip("第二个LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath2 = "";

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

    [Tooltip("第一个LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale = 0.85f;

    [Tooltip("第二个LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale2 = 0.5f;

    [Header("图像预处理参数")]
    [Tooltip("亮度调整")]
    [Range(0.5f, 3f)]
    public float _brightness = 1.0f;

    [Tooltip("对比度调整")]
    [Range(0.5f, 1.5f)]
    public float _contrast = 1.0f;

    [Tooltip("饱和度调整")]
    [Range(0.0f, 2.0f)]
    public float _saturation = 1.0f;

    [Space]
    [Tooltip("是否显示Python控制台")]
    public bool _showPythonConsole = false;

    [Tooltip("图像生成强度")]
    [Range(1f, 3.0f)]
    public float _strength = 1.0f;

    [Header("高级参数")]
    [Tooltip("Delta参数，控制噪声添加量")]
    [Range(0.1f, 1.0f)]
    public float _delta = 0.8f;

    [Tooltip("是否在每步添加噪声")]
    public bool _doAddNoise = true;

    [Tooltip("是否启用相似图像过滤")]
    public bool _enableSimilarFilter = true;

    [Tooltip("相似图像过滤阈值")]
    [Range(0.1f, 0.99f)]
    public float _similarThreshold = 0.6f;

    [Tooltip("最大跳过帧数")]
    [Range(1, 30)]
    public int _maxSkipFrame = 10;

    [Tooltip("引导尺度")]
    [Range(0.1f, 10.0f)]
    public float _guidanceScale = 1.0f;

    [Space]
    [Tooltip("绕过模式，直接返回输入图像而不经过AI处理")]
    public bool _bypassMode = false;

    [Space]
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
    private bool _isRunning = false, _isAdvancing = false;
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
                Debug.Log($"接收到完整图像数据，大小: {_advancedData.Length} 字节");
                if (_resultTexture == null)
                {
                    _resultTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false,
                                                   QualitySettings.activeColorSpace == ColorSpace.Linear);
                    //Debug.Log($"创建新的结果纹理，颜色空间：{QualitySettings.activeColorSpace}");
                    if (_resultMaterial != null)
                    {
                        _resultMaterial.mainTexture = _resultTexture;
                        Debug.Log("将新纹理分配给材质");
                    }
                    else
                        Debug.LogError("结果材质为空，无法分配纹理");
                }

                try
                {
                    // 记录图像数据的前20个字节，用于调试
                    string dataPrefix = "";
                    for (int i = 0; i < System.Math.Min(20, _advancedData.Length); i++)
                        { dataPrefix += _advancedData[i].ToString("X2") + " "; }
                    Debug.Log($"图像数据前缀: {dataPrefix}");
                    
                    // 验证图像格式
                    bool isPng = (_advancedData.Length > 8 && 
                        _advancedData[0] == 0x89 && _advancedData[1] == 0x50 && 
                        _advancedData[2] == 0x4E && _advancedData[3] == 0x47);
                    
                    bool isJpeg = (_advancedData.Length > 3 && 
                        _advancedData[0] == 0xFF && _advancedData[1] == 0xD8 && 
                        _advancedData[2] == 0xFF);
                    
                    //Debug.Log($"图像格式: {(isPng ? "PNG" : (isJpeg ? "JPEG" : "未知"))}");
                    bool success = _resultTexture.LoadImage(_advancedData, !isPng); // 如果不是PNG，保留透明度
                    Debug.Log($"加载图像{(success ? "成功" : "失败")}，宽度={_resultTexture.width}, 高度={_resultTexture.height}");
                    
                    if (success)
                    {
                        _resultTexture.Apply();
                        
                        // 检查图像的颜色
                        Color[] pixels = _resultTexture.GetPixels();
                        if (pixels.Length > 0)
                        {
                            float avgR = 0, avgG = 0, avgB = 0, avgA = 0;
                            for (int i = 0; i < pixels.Length; i++)
                            {
                                avgR += pixels[i].r;
                                avgG += pixels[i].g;
                                avgB += pixels[i].b;
                                avgA += pixels[i].a;
                            }
                            
                            avgR /= pixels.Length;
                            avgG /= pixels.Length;
                            avgB /= pixels.Length;
                            avgA /= pixels.Length;
                            //Debug.Log($"收到图像平均RGB值: R={avgR:F3}, G={avgG:F3}, B={avgB:F3}, A={avgA:F3}");
                            
                            // 检查是否是全黑图像
                            if (avgR < 0.01f && avgG < 0.01f && avgB < 0.01f)
                            {
                                Debug.LogWarning("检测到全黑图像！创建测试色彩图案");
                                
                                // 创建一个测试图案替换黑色图像
                                _resultTexture = new Texture2D(512, 512, TextureFormat.RGBA32, false,
                                                               QualitySettings.activeColorSpace == ColorSpace.Linear);
                                Color[] testPixels = new Color[512 * 512];
                                for (int y = 0; y < 512; y++)
                                {
                                    for (int x = 0; x < 512; x++)
                                    {
                                        if (x < 170)
                                            testPixels[y * 512 + x] = new Color(1, 0, 0, 1); // 红色区域
                                        else if (x < 340)
                                            testPixels[y * 512 + x] = new Color(0, 1, 0, 1); // 绿色区域
                                        else
                                            testPixels[y * 512 + x] = new Color(0, 0, 1, 1); // 蓝色区域
                                    }
                                }
                                _resultTexture.SetPixels(testPixels);
                                _resultTexture.Apply();
                                
                                if (_resultMaterial != null)
                                    _resultMaterial.mainTexture = _resultTexture;
                            }
                        }
                        else
                            Debug.LogError("纹理像素数组为空");
                    }
                    else
                    {
                        Debug.LogError("图像加载到纹理失败");
                        _resultTexture = new Texture2D(512, 512, TextureFormat.RGBA32, false,
                                                       QualitySettings.activeColorSpace == ColorSpace.Linear);

                        // 创建一个应急图像
                        Color[] testPixels = new Color[512 * 512];
                        for (int i = 0; i < testPixels.Length; i++)
                            testPixels[i] = new Color(1, 1, 0, 1); // 黄色
                        _resultTexture.SetPixels(testPixels);
                        _resultTexture.Apply();
                        if (_resultMaterial != null)
                            _resultMaterial.mainTexture = _resultTexture;
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"处理图像数据时出错: {e.Message}");
                }

                _advancedData = null;
                _isAdvancing = false;
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

    public byte[] PreprocessImage(byte[] imageBytes, int width, int height)
    {
        if (imageBytes == null || imageBytes.Length == 0)
            return imageBytes;
        
        // 创建纹理用于处理
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color32[] pixels = new Color32[width * height];
        
        // 将字节数组转换为Color32数组
        for (int i = 0; i < width * height; i++)
        {
            int byteIndex = i * 3;
            if (byteIndex + 2 < imageBytes.Length)
            {
                pixels[i] = new Color32(
                    imageBytes[byteIndex],
                    imageBytes[byteIndex + 1],
                    imageBytes[byteIndex + 2],
                    255);
            }
        }
        
        // 应用预处理调整
        for (int i = 0; i < pixels.Length; i++)
        {
            // 转换到线性空间进行处理
            Color linearColor = new Color(
                pixels[i].r / 255f,
                pixels[i].g / 255f,
                pixels[i].b / 255f);
            
            // 亮度调整
            linearColor *= _brightness;
            
            // 对比度调整 (先转到-0.5到0.5范围)
            if (_contrast != 1f)
            {
                linearColor.r = (linearColor.r - 0.5f) * _contrast + 0.5f;
                linearColor.g = (linearColor.g - 0.5f) * _contrast + 0.5f;
                linearColor.b = (linearColor.b - 0.5f) * _contrast + 0.5f;
            }
            
            // 饱和度调整
            if (_saturation != 1f)
            {
                float luminance = linearColor.r * 0.3f + linearColor.g * 0.59f + linearColor.b * 0.11f;
                linearColor.r = Mathf.Lerp(luminance, linearColor.r, _saturation);
                linearColor.g = Mathf.Lerp(luminance, linearColor.g, _saturation);
                linearColor.b = Mathf.Lerp(luminance, linearColor.b, _saturation);
            }
            
            // 确保值在合法范围内
            linearColor.r = Mathf.Clamp01(linearColor.r);
            linearColor.g = Mathf.Clamp01(linearColor.g);
            linearColor.b = Mathf.Clamp01(linearColor.b);
            
            // 转回Color32格式
            pixels[i] = new Color32(
                (byte)(linearColor.r * 255),
                (byte)(linearColor.g * 255),
                (byte)(linearColor.b * 255),
                255);
        }
        
        // 转回字节数组
        byte[] result = new byte[width * height * 3];
        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            result[byteIndex] = pixels[i].r;
            result[byteIndex + 1] = pixels[i].g;
            result[byteIndex + 2] = pixels[i].b;
        }
        return result;
    }

    public void setModelPaths()
    {
        // 构建完整路径
        string fullBaseModelPath = GetFullModelPath(_baseModelPath);
        string fullVaePath = GetFullModelPath(_tinyVaeModelPath);
        
        // 修复：即使是空路径也传递空字符串，而不是null
        string fullLoraPath = string.IsNullOrEmpty(_loraModelPath) 
            ? "" 
            : GetFullModelPath(_loraModelPath);
        
        string fullLoraPath2 = string.IsNullOrEmpty(_loraModelPath2)
            ? "" : GetFullModelPath(_loraModelPath2);

        Debug.Log(
            $"Sending paths to Python server: Base={fullBaseModelPath}, VAE={fullVaePath}, LoRA1={fullLoraPath}, LoRA2={fullLoraPath2}"
        );

        // 恢复使用原始参数名lora_model和lora_model2以匹配Python端
        string cmd0 =
            $"|start|command||paths||base_model||{fullBaseModelPath}||taesd_model||{fullVaePath}"
            + $"||lora_model||{fullLoraPath}||lora_model2||{fullLoraPath2}||run||0|end|";
        SendCommandToPython(cmd0);
    }

    public void LoadPipeline()
    {
        int vae = (_useTinyVae ? 1 : 0),
            lora = (_useLcmLora ? 1 : 0);
        Debug.Log(
            $"准备发送参数到Python: 宽度={_width}, 高度={_height}, 种子={_seed}, 强度={_strength}, LoRA1强度={_loraScale}, LoRA2强度={_loraScale2}"
        );
        Debug.Log(
            $"高级参数: Delta={_delta}, 添加噪声={_doAddNoise}, 相似图像过滤={_enableSimilarFilter}, 阈值={_similarThreshold}, 最大跳帧={_maxSkipFrame}"
        );
        Debug.Log(
            $"引导尺度: {_guidanceScale}, 绕过模式: {_bypassMode}"
        );

        // 确保种子值作为字符串发送
        string seedStr = _seed.ToString();

        // 恢复使用原始参数名lora_scale和lora_scale2，添加新参数
        string loadCmd =
            $"|start|command||load||width||{_width}||height||{_height}||seed||{seedStr}"
            + $"||use_vae||{vae}||use_lora||{lora}||strength||{_strength}||lora_scale||{_loraScale}||lora_scale2||{_loraScale2}"
            + $"||delta||{_delta}||do_add_noise||{(_doAddNoise ? 1 : 0)}||enable_similar_filter||{(_enableSimilarFilter ? 1 : 0)}"
            + $"||similar_threshold||{_similarThreshold}||max_skip_frame||{_maxSkipFrame}||guidance_scale||{_guidanceScale}||acceleration||{_acceleration}"
            + $"||bypass_mode||{(_bypassMode ? "true" : "false")}"
            + $"||prompt||{_defaultPrompt}||neg_prompt||{_defaultNegativePrompt}||run||0|end|";

        //Debug.Log($"发送命令长度: {loadCmd.Length} 字节");
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
                //Debug.Log($"Command sent, length: {messageBytes.Length} bytes");
            }
            else
                Debug.LogError("Cannot send command - stream is null or not writable");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending command to Python: {e.Message}");
        }
    }

    // 扩展AdvancePipeline方法，支持动态传递强度参数和图像预处理
    public void AdvancePipeline(
        Texture2D tex,
        string prompt,
        float? strength = null,
        float? loraScale = null,
        float? loraScale2 = null
    )
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
            float currentLoraScale2 = loraScale2 ?? _loraScale2;

            Debug.Log(
                $"动态生成: 提示词='{prompt}', 强度={currentStrength}, LoRA1强度={currentLoraScale}, LoRA2强度={currentLoraScale2}, 绕过模式={_bypassMode}"
            );
            
            // 记录当前颜色空间
            bool isLinearSpace = QualitySettings.activeColorSpace == ColorSpace.Linear;
            Debug.Log($"当前颜色空间: {(isLinearSpace ? "线性" : "Gamma")}");
            
            // 记录原始图像的RGB均值
            Color[] pixels = tex.GetPixels();
            float avgR = 0, avgG = 0, avgB = 0;
            foreach (Color p in pixels)
            {
                avgR += p.r;
                avgG += p.g;
                avgB += p.b;
            }
            if (pixels.Length > 0)
            {
                avgR /= pixels.Length;
                avgG /= pixels.Length;
                avgB /= pixels.Length;
                Debug.Log($"原始图像平均RGB值: R={avgR:F2}, G={avgG:F2}, B={avgB:F2}");
            }
            
            // 改用PNG格式避免JPEG压缩导致的颜色损失
            byte[] imageBytes = ImageConversion.EncodeToPNG(tex);
            
            if (imageBytes == null || imageBytes.Length == 0)
            {
                Debug.LogError("Failed to encode image to PNG");
                return;
            }
            
            // 使用Base64编码处理图像数据，避免二进制数据中的特殊字符干扰协议解析
            string base64Image = System.Convert.ToBase64String(imageBytes);
            Debug.Log(
                $"使用PNG编码，大小: {imageBytes.Length} 字节, Base64长度: {base64Image.Length}"
            );

            // 添加颜色空间信息到命令中
            string command =
                $"|start|command||advance||prompt||{prompt}||strength||{currentStrength}||lora_scale||{currentLoraScale}||lora_scale2||{currentLoraScale2}||is_linear_space||{isLinearSpace.ToString().ToLower()}||bypass_mode||{_bypassMode.ToString().ToLower()}||image_base64||{base64Image}||run||0|end|";

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

    // 添加InputImage方法
    public void InputImage(byte[] imageBytes, int width, int height)
    {
        if (!isRunning() || imageBytes == null || imageBytes.Length != width * height * 3)
        {
            Debug.LogError($"InputImage失败: 服务未运行或图像数据无效");
            return;
        }

        try
        {
            // 创建Texture2D并设置像素数据
            Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(imageBytes);
            tex.Apply();
            
            // 使用默认参数调用AdvancePipeline
            AdvancePipeline(
                tex, 
                _defaultPrompt
            );
            
            // 清理临时纹理
            Destroy(tex);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"InputImage处理错误: {e.Message}");
        }
    }
}
*/

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
    public string _baseModelPath = "kohaku-v2.1";

    [Tooltip("VAE模型路径，相对于StreamingAssets/models目录")]
    public string _tinyVaeModelPath = "taesd";

    [Tooltip("第一个LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath = "lcm-lora-sdv1-5";

    [Tooltip("第二个LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath2 = "";

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

    [Tooltip("第一个LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale = 0.85f;

    [Tooltip("第二个LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale2 = 0.5f;

    [Header("图像预处理参数")]
    [Tooltip("亮度调整")]
    [Range(0.5f, 3f)]
    public float _brightness = 1.0f;

    [Tooltip("对比度调整")]
    [Range(0.5f, 1.5f)]
    public float _contrast = 1.0f;

    [Tooltip("饱和度调整")]
    [Range(0.0f, 2.0f)]
    public float _saturation = 1.0f;

    [Space]
    [Tooltip("是否显示Python控制台")]
    public bool _showPythonConsole = false;

    [Tooltip("图像生成强度")]
    [Range(1f, 3.0f)]
    public float _strength = 1.0f;

    [Header("高级参数")]
    [Tooltip("Delta参数，控制噪声添加量")]
    [Range(0.1f, 1.0f)]
    public float _delta = 0.8f;

    [Tooltip("是否在每步添加噪声")]
    public bool _doAddNoise = true;

    [Tooltip("是否启用相似图像过滤")]
    public bool _enableSimilarFilter = true;

    [Tooltip("相似图像过滤阈值")]
    [Range(0.1f, 0.99f)]
    public float _similarThreshold = 0.6f;

    [Tooltip("最大跳过帧数")]
    [Range(1, 30)]
    public int _maxSkipFrame = 10;

    [Tooltip("引导尺度")]
    [Range(0.1f, 10.0f)]
    public float _guidanceScale = 1.0f;

    [Space]
    [Tooltip("绕过模式，直接返回输入图像而不经过AI处理")]
    public bool _bypassMode = false;

    [Space]
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
    private bool _isRunning = false, _isAdvancing = false;
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
                Debug.Log($"接收到完整图像数据，大小: {_advancedData.Length} 字节");
                if (_resultTexture == null)
                {
                    _resultTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false,
                                                   QualitySettings.activeColorSpace == ColorSpace.Linear);
                    //Debug.Log($"创建新的结果纹理，颜色空间：{QualitySettings.activeColorSpace}");
                    if (_resultMaterial != null)
                    {
                        _resultMaterial.mainTexture = _resultTexture;
                        Debug.Log("将新纹理分配给材质");
                    }
                    else
                        Debug.LogError("结果材质为空，无法分配纹理");
                }

                try
                {
                    // 记录图像数据的前20个字节，用于调试
                    string dataPrefix = "";
                    for (int i = 0; i < System.Math.Min(20, _advancedData.Length); i++)
                    { dataPrefix += _advancedData[i].ToString("X2") + " "; }
                    Debug.Log($"图像数据前缀: {dataPrefix}");

                    // 验证图像格式
                    bool isPng = (_advancedData.Length > 8 &&
                        _advancedData[0] == 0x89 && _advancedData[1] == 0x50 &&
                        _advancedData[2] == 0x4E && _advancedData[3] == 0x47);

                    bool isJpeg = (_advancedData.Length > 3 &&
                        _advancedData[0] == 0xFF && _advancedData[1] == 0xD8 &&
                        _advancedData[2] == 0xFF);

                    //Debug.Log($"图像格式: {(isPng ? "PNG" : (isJpeg ? "JPEG" : "未知"))}");
                    bool success = _resultTexture.LoadImage(_advancedData, !isPng); // 如果不是PNG，保留透明度
                    Debug.Log($"加载图像{(success ? "成功" : "失败")}，宽度={_resultTexture.width}, 高度={_resultTexture.height}");

                    if (success)
                    {
                        _resultTexture.Apply();

                        // 检查图像的颜色
                        Color[] pixels = _resultTexture.GetPixels();
                        if (pixels.Length > 0)
                        {
                            float avgR = 0, avgG = 0, avgB = 0, avgA = 0;
                            for (int i = 0; i < pixels.Length; i++)
                            {
                                avgR += pixels[i].r;
                                avgG += pixels[i].g;
                                avgB += pixels[i].b;
                                avgA += pixels[i].a;
                            }

                            avgR /= pixels.Length;
                            avgG /= pixels.Length;
                            avgB /= pixels.Length;
                            avgA /= pixels.Length;
                            //Debug.Log($"收到图像平均RGB值: R={avgR:F3}, G={avgG:F3}, B={avgB:F3}, A={avgA:F3}");

                            // 检查是否是全黑图像
                            if (avgR < 0.01f && avgG < 0.01f && avgB < 0.01f)
                            {
                                Debug.LogWarning("检测到全黑图像！创建测试色彩图案");

                                // 创建一个测试图案替换黑色图像
                                _resultTexture = new Texture2D(512, 512, TextureFormat.RGBA32, false,
                                                               QualitySettings.activeColorSpace == ColorSpace.Linear);
                                Color[] testPixels = new Color[512 * 512];
                                for (int y = 0; y < 512; y++)
                                {
                                    for (int x = 0; x < 512; x++)
                                    {
                                        if (x < 170)
                                            testPixels[y * 512 + x] = new Color(1, 0, 0, 1); // 红色区域
                                        else if (x < 340)
                                            testPixels[y * 512 + x] = new Color(0, 1, 0, 1); // 绿色区域
                                        else
                                            testPixels[y * 512 + x] = new Color(0, 0, 1, 1); // 蓝色区域
                                    }
                                }
                                _resultTexture.SetPixels(testPixels);
                                _resultTexture.Apply();

                                if (_resultMaterial != null)
                                    _resultMaterial.mainTexture = _resultTexture;
                            }
                        }
                        else
                            Debug.LogError("纹理像素数组为空");
                    }
                    else
                    {
                        Debug.LogError("图像加载到纹理失败");
                        _resultTexture = new Texture2D(512, 512, TextureFormat.RGBA32, false,
                                                       QualitySettings.activeColorSpace == ColorSpace.Linear);

                        // 创建一个应急图像
                        Color[] testPixels = new Color[512 * 512];
                        for (int i = 0; i < testPixels.Length; i++)
                            testPixels[i] = new Color(1, 1, 0, 1); // 黄色
                        _resultTexture.SetPixels(testPixels);
                        _resultTexture.Apply();
                        if (_resultMaterial != null)
                            _resultMaterial.mainTexture = _resultTexture;
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"处理图像数据时出错: {e.Message}");
                }

                _advancedData = null;
                _isAdvancing = false;
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

    public byte[] PreprocessImage(byte[] imageBytes, int width, int height)
    {
        if (imageBytes == null || imageBytes.Length == 0)
            return imageBytes;

        // 创建纹理用于处理
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color32[] pixels = new Color32[width * height];

        // 将字节数组转换为Color32数组
        for (int i = 0; i < width * height; i++)
        {
            int byteIndex = i * 3;
            if (byteIndex + 2 < imageBytes.Length)
            {
                pixels[i] = new Color32(
                    imageBytes[byteIndex],
                    imageBytes[byteIndex + 1],
                    imageBytes[byteIndex + 2],
                    255);
            }
        }

        // 应用预处理调整
        for (int i = 0; i < pixels.Length; i++)
        {
            // 转换到线性空间进行处理
            Color linearColor = new Color(
                pixels[i].r / 255f,
                pixels[i].g / 255f,
                pixels[i].b / 255f);

            // 亮度调整
            linearColor *= _brightness;

            // 对比度调整 (先转到-0.5到0.5范围)
            if (_contrast != 1f)
            {
                linearColor.r = (linearColor.r - 0.5f) * _contrast + 0.5f;
                linearColor.g = (linearColor.g - 0.5f) * _contrast + 0.5f;
                linearColor.b = (linearColor.b - 0.5f) * _contrast + 0.5f;
            }

            // 饱和度调整
            if (_saturation != 1f)
            {
                float luminance = linearColor.r * 0.3f + linearColor.g * 0.59f + linearColor.b * 0.11f;
                linearColor.r = Mathf.Lerp(luminance, linearColor.r, _saturation);
                linearColor.g = Mathf.Lerp(luminance, linearColor.g, _saturation);
                linearColor.b = Mathf.Lerp(luminance, linearColor.b, _saturation);
            }

            // 确保值在合法范围内
            linearColor.r = Mathf.Clamp01(linearColor.r);
            linearColor.g = Mathf.Clamp01(linearColor.g);
            linearColor.b = Mathf.Clamp01(linearColor.b);

            // 转回Color32格式
            pixels[i] = new Color32(
                (byte)(linearColor.r * 255),
                (byte)(linearColor.g * 255),
                (byte)(linearColor.b * 255),
                255);
        }

        // 转回字节数组
        byte[] result = new byte[width * height * 3];
        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            result[byteIndex] = pixels[i].r;
            result[byteIndex + 1] = pixels[i].g;
            result[byteIndex + 2] = pixels[i].b;
        }
        return result;
    }

    public void setModelPaths()
    {
        // 构建完整路径
        string fullBaseModelPath = GetFullModelPath(_baseModelPath);
        string fullVaePath = GetFullModelPath(_tinyVaeModelPath);

        // 修复：即使是空路径也传递空字符串，而不是null
        string fullLoraPath = string.IsNullOrEmpty(_loraModelPath)
            ? ""
            : GetFullModelPath(_loraModelPath);

        string fullLoraPath2 = string.IsNullOrEmpty(_loraModelPath2)
            ? "" : GetFullModelPath(_loraModelPath2);

        Debug.Log(
            $"Sending paths to Python server: Base={fullBaseModelPath}, VAE={fullVaePath}, LoRA1={fullLoraPath}, LoRA2={fullLoraPath2}"
        );

        // 恢复使用原始参数名lora_model和lora_model2以匹配Python端
        string cmd0 =
            $"|start|command||paths||base_model||{fullBaseModelPath}||taesd_model||{fullVaePath}"
            + $"||lora_model||{fullLoraPath}||lora_model2||{fullLoraPath2}||run||0|end|";
        SendCommandToPython(cmd0);
    }

    public void LoadPipeline()
    {
        int vae = (_useTinyVae ? 1 : 0),
            lora = (_useLcmLora ? 1 : 0);
        Debug.Log(
            $"准备发送参数到Python: 宽度={_width}, 高度={_height}, 种子={_seed}, 强度={_strength}, LoRA1强度={_loraScale}, LoRA2强度={_loraScale2}"
        );
        Debug.Log(
            $"高级参数: Delta={_delta}, 添加噪声={_doAddNoise}, 相似图像过滤={_enableSimilarFilter}, 阈值={_similarThreshold}, 最大跳帧={_maxSkipFrame}"
        );
        Debug.Log(
            $"引导尺度: {_guidanceScale}, 绕过模式: {_bypassMode}"
        );

        // 确保种子值作为字符串发送
        string seedStr = _seed.ToString();

        // 恢复使用原始参数名lora_scale和lora_scale2，添加新参数
        string loadCmd =
            $"|start|command||load||width||{_width}||height||{_height}||seed||{seedStr}"
            + $"||use_vae||{vae}||use_lora||{lora}||strength||{_strength}||lora_scale||{_loraScale}||lora_scale2||{_loraScale2}"
            + $"||delta||{_delta}||do_add_noise||{(_doAddNoise ? 1 : 0)}||enable_similar_filter||{(_enableSimilarFilter ? 1 : 0)}"
            + $"||similar_threshold||{_similarThreshold}||max_skip_frame||{_maxSkipFrame}||guidance_scale||{_guidanceScale}||acceleration||{_acceleration}"
            + $"||bypass_mode||{(_bypassMode ? "true" : "false")}"
            + $"||prompt||{_defaultPrompt}||neg_prompt||{_defaultNegativePrompt}||run||0|end|";

        //Debug.Log($"发送命令长度: {loadCmd.Length} 字节");
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
                //Debug.Log($"Command sent, length: {messageBytes.Length} bytes");
            }
            else
                Debug.LogError("Cannot send command - stream is null or not writable");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending command to Python: {e.Message}");
        }
    }

    // 扩展AdvancePipeline方法，支持动态传递强度参数和图像预处理
    /* public void AdvancePipeline(
         Texture2D tex,
         string prompt,
         float? strength = null,
         float? loraScale = null,
         float? loraScale2 = null
     )
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
             float currentLoraScale2 = loraScale2 ?? _loraScale2;

             Debug.Log(
                 $"动态生成: 提示词='{prompt}', 强度={currentStrength}, LoRA1强度={currentLoraScale}, LoRA2强度={currentLoraScale2}, 绕过模式={_bypassMode}"
             );

             // 记录当前颜色空间
             bool isLinearSpace = QualitySettings.activeColorSpace == ColorSpace.Linear;
             Debug.Log($"当前颜色空间: {(isLinearSpace ? "线性" : "Gamma")}");

             // 记录原始图像的RGB均值
             Color[] pixels = tex.GetPixels();
             float avgR = 0, avgG = 0, avgB = 0;
             foreach (Color p in pixels)
             {
                 avgR += p.r;
                 avgG += p.g;
                 avgB += p.b;
             }
             if (pixels.Length > 0)
             {
                 avgR /= pixels.Length;
                 avgG /= pixels.Length;
                 avgB /= pixels.Length;
                 Debug.Log($"原始图像平均RGB值: R={avgR:F2}, G={avgG:F2}, B={avgB:F2}");
             }

             // 改用PNG格式避免JPEG压缩导致的颜色损失
             byte[] imageBytes = ImageConversion.EncodeToPNG(tex);

             if (imageBytes == null || imageBytes.Length == 0)
             {
                 Debug.LogError("Failed to encode image to PNG");
                 return;
             }

             // 使用Base64编码处理图像数据，避免二进制数据中的特殊字符干扰协议解析
             string base64Image = System.Convert.ToBase64String(imageBytes);
             Debug.Log(
                 $"使用PNG编码，大小: {imageBytes.Length} 字节, Base64长度: {base64Image.Length}"
             );

             // 添加颜色空间信息到命令中
             string command =
                 $"|start|command||advance||prompt||{prompt}||strength||{currentStrength}||lora_scale||{currentLoraScale}||lora_scale2||{currentLoraScale2}||is_linear_space||{isLinearSpace.ToString().ToLower()}||bypass_mode||{_bypassMode.ToString().ToLower()}||image_base64||{base64Image}||run||0|end|";

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
 */

    // StreamDiffusionClient.cs의 AdvancePipeline 메서드 수정
    public void AdvancePipeline(
        Texture2D tex,
        string prompt,
        float? strength = null,
        float? loraScale = null,
        float? loraScale2 = null
    )
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

            // 사용할 파라미터 설정
            float currentStrength = strength ?? _strength;
            float currentLoraScale = loraScale ?? _loraScale;
            float currentLoraScale2 = loraScale2 ?? _loraScale2;

            Debug.Log(
                $"동적생성: 제시어='{prompt}', 강도={currentStrength}, LoRA1강도={currentLoraScale}, LoRA2강도={currentLoraScale2}, 우회모드={_bypassMode}"
            );

            // 컬러 스페이스 확인
            bool isLinearSpace = QualitySettings.activeColorSpace == ColorSpace.Linear;
            Debug.Log($"현재 컬러 스페이스: {(isLinearSpace ? "선형" : "Gamma")}");

            // 텍스처 포맷 호환성 확인 및 변환
            Texture2D processTexture = tex;
            bool needsConversion = false;

            // 압축된 포맷이거나 EncodeToPNG를 지원하지 않는 포맷인지 확인
            if (tex.format == TextureFormat.DXT1 ||
                tex.format == TextureFormat.DXT5 ||
                tex.format == TextureFormat.ETC_RGB4 ||
                tex.format == TextureFormat.ETC2_RGBA8 ||
                tex.format == TextureFormat.ASTC_4x4 ||
                tex.format == TextureFormat.ASTC_6x6 ||
                tex.format == TextureFormat.ASTC_8x8 ||
                tex.format == TextureFormat.ASTC_10x10 ||
                tex.format == TextureFormat.ASTC_12x12 ||
                tex.format == TextureFormat.PVRTC_RGB2 ||
                tex.format == TextureFormat.PVRTC_RGBA2 ||
                tex.format == TextureFormat.PVRTC_RGB4 ||
                tex.format == TextureFormat.PVRTC_RGBA4 ||
                !tex.isReadable)
            {
                needsConversion = true;
                Debug.LogWarning($"텍스처 포맷 {tex.format}은 EncodeToPNG를 지원하지 않습니다. 변환합니다.");
            }

            if (needsConversion)
            {
                // RenderTexture를 통해 호환 가능한 포맷으로 변환
                RenderTexture renderTexture = RenderTexture.GetTemporary(
                    tex.width,
                    tex.height,
                    0,
                    RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Default
                );

                // 원본 텍스처를 RenderTexture에 복사
                Graphics.Blit(tex, renderTexture);

                // 새로운 Texture2D 생성 (PNG 인코딩 호환 포맷)
                processTexture = new Texture2D(
                    tex.width,
                    tex.height,
                    TextureFormat.RGBA32,
                    false,
                    isLinearSpace
                );

                // RenderTexture에서 픽셀 읽기
                RenderTexture previousActive = RenderTexture.active;
                RenderTexture.active = renderTexture;

                processTexture.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
                processTexture.Apply();

                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(renderTexture);
            }

            // 원본 이미지의 RGB 평균값 기록
            Color[] pixels = processTexture.GetPixels();
            float avgR = 0, avgG = 0, avgB = 0;
            foreach (Color p in pixels)
            {
                avgR += p.r;
                avgG += p.g;
                avgB += p.b;
            }
            if (pixels.Length > 0)
            {
                avgR /= pixels.Length;
                avgG /= pixels.Length;
                avgB /= pixels.Length;
                Debug.Log($"원본 이미지 평균 RGB값: R={avgR:F2}, G={avgG:F2}, B={avgB:F2}");
            }

            // PNG로 인코딩
            byte[] imageBytes = ImageConversion.EncodeToPNG(processTexture);

            // 변환된 텍스처가 있다면 정리
            if (needsConversion && processTexture != tex)
            {
                Destroy(processTexture);
            }

            if (imageBytes == null || imageBytes.Length == 0)
            {
                Debug.LogError("PNG 인코딩 실패");
                return;
            }

            // Base64 인코딩으로 이미지 데이터 처리
            string base64Image = System.Convert.ToBase64String(imageBytes);
            Debug.Log(
                $"PNG 인코딩 사용, 크기: {imageBytes.Length} 바이트, Base64 길이: {base64Image.Length}"
            );

            // 컬러 스페이스 정보를 포함한 명령어 생성
            string command =
                $"|start|command||advance||prompt||{prompt}||strength||{currentStrength}||lora_scale||{currentLoraScale}||lora_scale2||{currentLoraScale2}||is_linear_space||{isLinearSpace.ToString().ToLower()}||bypass_mode||{_bypassMode.ToString().ToLower()}||image_base64||{base64Image}||run||0|end|";

            // 큰 데이터를 청크 단위로 전송
            byte[] commandBytes = Encoding.UTF8.GetBytes(command);
            int chunkSize = 16384; // 16KB 청크
            int sentBytes = 0;

            while (sentBytes < commandBytes.Length)
            {
                int remaining = commandBytes.Length - sentBytes;
                int currentChunkSize = System.Math.Min(remaining, chunkSize);

                _stream.Write(commandBytes, sentBytes, currentChunkSize);
                sentBytes += currentChunkSize;

                // 버퍼 오버플로우 방지를 위한 대기
                if (sentBytes < commandBytes.Length)
                {
                    Thread.Sleep(5);
                }
            }

            Debug.Log($"청크 단위로 전송 완료, 총 크기: {commandBytes.Length} 바이트");

            _isAdvancing = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"AdvancePipeline 에러: {e.Message}");
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

    // 添加InputImage方法
    public void InputImage(byte[] imageBytes, int width, int height)
    {
        if (!isRunning() || imageBytes == null || imageBytes.Length != width * height * 3)
        {
            Debug.LogError($"InputImage失败: 服务未运行或图像数据无效");
            return;
        }

        try
        {
            // 创建Texture2D并设置像素数据
            Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(imageBytes);
            tex.Apply();

            // 使用默认参数调用AdvancePipeline
            AdvancePipeline(
                tex,
                _defaultPrompt
            );

            // 清理临时纹理
            Destroy(tex);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"InputImage处理错误: {e.Message}");
        }
    }
}
