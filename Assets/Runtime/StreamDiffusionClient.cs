using PimDeWitte.UnityMainThreadDispatcher;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class StreamDiffusionClient : MonoBehaviour
{
    public string _baseModelPath = "./models/kohaku-v2.1";
    public string _tinyVaeModelPath = "./models/taesd";
    public string _loraModelPath = "./models/lcm-lora-sdv1-5";
    public string _acceleration = "tensorrt";

    public int _width = 512, _height = 512, _seed = 2;
    public bool _useTinyVae = true, _useLcmLora = true;
    public string _defaultPrompt = "1 girl with blue dog hair, smiling";
    public string _defaultNegativePrompt = "low quality, bad quality, blurry, low resolution";

    public Material _resultMaterial;
    private Texture2D _resultTexture;
    private System.Diagnostics.Process _backgroundProcess;

    private string _serverIP = "127.0.0.1";
    private int _serverPort = 9999;
    private int _pipelineLoaded = 0;
    private bool _isRunning = false, _isAdvancing = false;

    private TcpClient _client;
    private NetworkStream _stream;
    private Thread _clientThread;
    private byte[] _advancedData;

    public bool isValid() { return _isRunning; }
    public bool isGenerating() { return _isAdvancing; }
    public bool isPending() { return _isRunning && _pipelineLoaded < 0; }
    public bool isRunning() { return _isRunning && _pipelineLoaded > 0; }

    private void ConnectToServer()
    {
        try
        {
            _client.Connect(_serverIP, _serverPort);
            _stream = _client.GetStream();
            _isRunning = true;
            setModelPaths();

            Thread receiveThread = new Thread(new ThreadStart(ReceiveMessages));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (System.Exception e)
        { Debug.LogError("Error connecting to server: " + e.Message); }
    }

    private void ReceiveMessages()
    {
        try
        {
            while (_isRunning)
            {
                if (_stream.DataAvailable)
                {
                    byte[] receivedBytes = new byte[4096];
                    int bytesRead = _stream.Read(receivedBytes, 0, receivedBytes.Length);
                    //Debug.Log("Received from server: " + bytesRead);

                    UnityMainThreadDispatcher.Instance().Enqueue(() =>
                    { UpdateServerData(receivedBytes, bytesRead); });
                }
            }
        }
        catch (System.Exception e)
        { Debug.LogError("Error receiving messages: " + e.Message); }
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
                if (result == "loaded") _pipelineLoaded = 1;
                else _pipelineLoaded = 0;
            }
        }
        else if (length > 10)
        {
            byte[] lastFourBytes = new byte[4];
            System.Array.Copy(buffer, buffer.Length - 4, lastFourBytes, 0, 4);
            bool validEnd = true; byte pipeByte = (byte)'|';
            for (int i = 0; i < 4; i++) { if (lastFourBytes[i] != pipeByte) validEnd = false; }

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

                if (_resultTexture.LoadImage(_advancedData)) _resultTexture.Apply();
                _advancedData = null; _isAdvancing = false;
                //Debug.Log("Received image: " + len0);
            }
        }
        else
        {
            string result = Encoding.UTF8.GetString(buffer);
            _advancedData = null; _isAdvancing = false;
            Debug.Log("Received message: " + result);
        }
    }

    public void setModelPaths()
    {
        string cmd0 = $"|start|command||paths||base_model||{_baseModelPath}||taesd_model||{_tinyVaeModelPath}"
                    + $"||lora_model||{_loraModelPath}||run||0|end|";
        if (_stream != null && _stream.CanWrite)
        {
            byte[] messageBytes = Encoding.UTF8.GetBytes(cmd0);
            _stream.Write(messageBytes, 0, messageBytes.Length);
        }
    }

    public void LoadPipeline()
    {
        int vae = (_useTinyVae ? 1 : 0), lora = (_useLcmLora ? 1 : 0);
        string loadCmd = $"|start|command||load||width||{_width}||height||{_height}||seed||{_seed}"
                       + $"||use_vae||{vae}||use_lora||{lora}||acceleration||{_acceleration}"
                       + $"||prompt||{_defaultPrompt}||neg_prompt||${_defaultNegativePrompt}||run||0|end|";
        _pipelineLoaded = -1;  // pending
        if (_stream != null && _stream.CanWrite)
        {
            byte[] messageBytes = Encoding.UTF8.GetBytes(loadCmd);
            _stream.Write(messageBytes, 0, messageBytes.Length);
        }
    }

    public void AdvancePipeline(Texture2D tex, string prompt)
    {
        string cmd0 = $"|start|command||advance||prompt||{prompt}||image||", cmd1 = "||run||0|end|";
        byte[] bytes0 = Encoding.UTF8.GetBytes(cmd0), bytes1 = Encoding.UTF8.GetBytes(cmd1);
        byte[] bytes = ImageConversion.EncodeToJPG(tex);
        if (bytes == null) return;

        byte[] result = new byte[bytes0.Length + bytes.Length + bytes1.Length];
        System.Buffer.BlockCopy(bytes0, 0, result, 0, bytes0.Length);
        System.Buffer.BlockCopy(bytes, 0, result, bytes0.Length, bytes.Length);
        System.Buffer.BlockCopy(bytes1, 0, result, bytes0.Length + bytes.Length, bytes1.Length);
        if (_stream != null && _stream.CanWrite)
        {
            //Debug.Log("Request image: " + result.Length);
            _stream.Write(result, 0, result.Length);
            _isAdvancing = true;
        }
    }

    private void OutputDataReceived(object sender, System.Diagnostics.DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data)) Debug.Log(e.Data);
    }

    private void StartTcpClient()
    {
        Debug.Log("Start TCP client...");
        _client = new TcpClient();
        _clientThread = new Thread(new ThreadStart(ConnectToServer));
        _clientThread.IsBackground = true;
        _clientThread.Start();
    }

    void Start()
    {
        var pythonHome = $"{Application.streamingAssetsPath}/envs/streamdiffusion";
        var projectHome = $"{Application.streamingAssetsPath}/streamdiffusion";
        var scripts = $"{pythonHome}/Scripts";

        var path = System.Environment.GetEnvironmentVariable("PATH")?.TrimEnd(';');
        path = string.IsNullOrEmpty(path) ? $"{pythonHome};{scripts}" : $"{pythonHome};{scripts};{path}";
        System.Environment.SetEnvironmentVariable("PATH", path, System.EnvironmentVariableTarget.Process);

        var exePath = $"{pythonHome}/python.exe";
        if (System.IO.File.Exists(exePath))
        {
            Debug.Log("Start background server...");
            System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo(exePath)
            {
                FileName = exePath,
                Arguments = "image_predictor.py",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden,
                WorkingDirectory = $"{Application.streamingAssetsPath}"
            };

            _backgroundProcess = new System.Diagnostics.Process();
            _backgroundProcess.StartInfo = startInfo;
            _backgroundProcess.OutputDataReceived += OutputDataReceived;
            _backgroundProcess.ErrorDataReceived += OutputDataReceived;

            if (!_backgroundProcess.Start()) Debug.LogError("Failed to start the predictor");
            _backgroundProcess.BeginOutputReadLine();
            _backgroundProcess.BeginErrorReadLine();
        }
        else
            Debug.LogError("Failed to find Python.exe");
        Invoke("StartTcpClient", 8.0f);
    }

    void OnDestroy()
    {
        _isRunning = false; _pipelineLoaded = 0;
        if (_backgroundProcess != null && !_backgroundProcess.HasExited)
            _backgroundProcess.Kill();
        if (_clientThread != null) _clientThread.Interrupt();
        if (_client != null) _client.Close();
    }
}
