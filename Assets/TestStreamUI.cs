using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using PaintIn2D;  // 引入PaintIn2D命名空间

public class TestStreamUI : MonoBehaviour
{
    public StreamDiffusionClient _stream;
    public Material _inputMaterial;
    public Button _startButton;
    public InputField _promptInput;
    
    // 添加绘画相关组件引用
    [Header("绘画设置")]
    public CwPaintableSprite _paintableSprite; // 可绘制的Sprite组件
    public CwPaintableSpriteTexture _paintableTexture; // 可绘制的纹理组件
    public bool _usePaintTexture = true; // 是否使用绘制的纹理作为输入
    
    [Header("连续生成设置")]
    [Tooltip("启用后会自动按照指定间隔持续推送数据到StreamDiffusion")]
    public bool _continuousGeneration = false; // 是否连续推送数据到StreamDiffusion
    [Tooltip("连续生成的时间间隔(秒)")]
    public float _generationInterval = 0.5f; // 生成间隔，单位秒
    
    [Header("绕过模式设置")]
    [Tooltip("启用后将直接返回输入图像，不经过AI处理")]
    public bool _bypassMode = false; // 是否启用绕过模式
    public Toggle _bypassModeToggle; // 绕过模式的UI切换控件
    
    private WebCamTexture _webcamTexture = null;
    private Texture2D _inputTexture = null;
    private Texture _originalInputTexture = null;
    private bool _lastRunningStatus = false;
    private Texture2D _paintTexture = null; // 存储绘制的纹理
    private Coroutine _continuousGenerationCoroutine = null; // 持续生成协程
    private bool _previousContinuousGeneration = false; // 记录之前的连续生成状态
    private byte[] _imageBytes = null;

    public void StartWebcam()
    {
        if (_usePaintTexture) return; // 如果使用绘制模式，不启动网络摄像头
        
        WebCamDevice[] devices = WebCamTexture.devices;
        if (_webcamTexture != null || devices.Length == 0) return;

        string deviceName = devices[0].name;
        _webcamTexture = new WebCamTexture(deviceName);
        _webcamTexture.Play();
        StartCoroutine(UpdateWebcamData());
    }
    
    public void StartStreamDiff()
    {
        _startButton.interactable = false;
        
        // 将绕过模式设置传递给StreamDiffusionClient
        if (_stream != null)
        {
            _stream._bypassMode = _bypassMode;
        }
        
        if (_stream.isValid() && !_stream.isRunning()) _stream.LoadPipeline();
    }

    public void UpdateStreamDiff()
    {
        if (_stream.isRunning() && !_stream.isGenerating())
        {
            // 如果使用绘制的纹理，则获取当前绘制的纹理
            if (_usePaintTexture && _paintableTexture != null)
            {
                // 获取当前绘制的纹理
                UpdatePaintTexture();
            }
            
            _stream.AdvancePipeline(_inputTexture, _promptInput.text);
        }
    }
    
    // 连续生成协程
    private IEnumerator ContinuousGenerationRoutine()
    {
        Debug.Log("启动连续生成模式，间隔: " + _generationInterval + "秒");
        while (_continuousGeneration)
        {
            if (_stream.isRunning() && !_stream.isGenerating())
            {
                // 如果使用绘制的纹理，则获取当前绘制的纹理
                if (_usePaintTexture && _paintableTexture != null)
                {
                    // 获取当前绘制的纹理
                    UpdatePaintTexture();
                }
                
                // 推送当前数据
                _stream.AdvancePipeline(_inputTexture, _promptInput.text);
                Debug.Log("已推送数据进行生成");
            }
            
            // 等待指定时间
            yield return new WaitForSeconds(_generationInterval);
        }
        Debug.Log("停止连续生成模式");
    }
    
    // 新增方法：更新绘制的纹理
    private void UpdatePaintTexture()
    {
        if (_paintableTexture == null || _paintableTexture.Current == null) return;
        
        // 获取当前绘制的纹理（RenderTexture类型）
        RenderTexture sourceTexture = _paintableTexture.Current as RenderTexture;
        if (sourceTexture == null) return;
        
        // 如果_paintTexture尚未初始化或尺寸不匹配，则创建新的
        if (_paintTexture == null || 
            _paintTexture.width != sourceTexture.width || 
            _paintTexture.height != sourceTexture.height)
        {
            if (_paintTexture != null) Destroy(_paintTexture);
            // 确保创建的纹理格式与颜色空间匹配
            bool linearColorSpace = QualitySettings.activeColorSpace == ColorSpace.Linear;
            _paintTexture = new Texture2D(sourceTexture.width, sourceTexture.height, TextureFormat.RGBA32, false, linearColorSpace);
        }
        
        // 保存当前的RenderTexture
        RenderTexture prevRT = RenderTexture.active;
        
        try
        {
            // 设置当前RenderTexture为源纹理
            RenderTexture.active = sourceTexture;
            
            // 从RenderTexture读取像素到Texture2D
            _paintTexture.ReadPixels(new Rect(0, 0, sourceTexture.width, sourceTexture.height), 0, 0);
            _paintTexture.Apply();
            
            // 根据颜色空间进行颜色转换
            if (QualitySettings.activeColorSpace == ColorSpace.Linear)
            {
                // 确保传递给Python的颜色在sRGB空间正确
                // 由于我们创建了一个linear空间的纹理，ReadPixels会自动转换
                Debug.Log("使用线性颜色空间处理纹理");
            }
            else
            {
                Debug.Log("使用Gamma颜色空间处理纹理");
            }
            
            // 更新输入纹理
            _inputTexture = _paintTexture;
            
            // 更新输入材质的纹理
            if (_inputMaterial != null)
            {
                _inputMaterial.mainTexture = _inputTexture;
            }
            
            Debug.Log($"已更新绘制纹理: {_inputTexture.width}x{_inputTexture.height}, 颜色空间: {QualitySettings.activeColorSpace}");
        }
        finally
        {
            // 恢复之前的RenderTexture
            RenderTexture.active = prevRT;
        }
    }

    private IEnumerator UpdateWebcamData()
    {
        if (_usePaintTexture) yield break; // 如果使用绘制模式，不更新网络摄像头
        
        _inputTexture = new Texture2D(
            _webcamTexture.width, _webcamTexture.height, TextureFormat.RGBA32, false);
        _inputMaterial.mainTexture = _inputTexture;

        while (_webcamTexture.isPlaying)
        {
            _inputTexture.SetPixels32(_webcamTexture.GetPixels32());
            _inputTexture.Apply();
            UpdateStreamDiff();
            yield return new WaitForEndOfFrame();
        }
    }

    void Start()
    {
        _originalInputTexture = _inputMaterial.mainTexture;
        
        // 初始化绕过模式Toggle
        if (_bypassModeToggle != null)
        {
            _bypassModeToggle.isOn = _bypassMode;
            _bypassModeToggle.onValueChanged.AddListener(ToggleBypassMode);
        }
        
        if (_usePaintTexture)
        {
            // 如果使用绘制的纹理，确保组件存在
            if (_paintableSprite == null)
            {
                _paintableSprite = GetComponentInChildren<CwPaintableSprite>();
                if (_paintableSprite == null)
                {
                    Debug.LogError("未找到CwPaintableSprite组件，请手动设置！");
                    _usePaintTexture = false;
                }
            }
            
            if (_paintableTexture == null && _paintableSprite != null)
            {
                _paintableTexture = _paintableSprite.GetComponentInChildren<CwPaintableSpriteTexture>();
                if (_paintableTexture == null)
                {
                    Debug.LogError("未找到CwPaintableSpriteTexture组件，请手动设置！");
                    _usePaintTexture = false;
                }
            }
            
            // 如果找到了可绘制组件，设置初始纹理
            if (_usePaintTexture && _paintableTexture != null && _paintableTexture.Current != null)
            {
                UpdatePaintTexture();
            }
            else
            {
                // 如果无法使用绘制的纹理，回退到普通模式
                _inputTexture = _originalInputTexture as Texture2D;
            }
            
            // 用于定期更新绘制的纹理
            if (_usePaintTexture)
            {
                StartCoroutine(UpdatePaintTextureRoutine());
            }
        }
        else
        {
            _inputTexture = _originalInputTexture as Texture2D;
        }
        
        // 记录初始连续生成状态
        _previousContinuousGeneration = _continuousGeneration;
        
        // 如果一开始就启用了连续生成，立即启动协程
        if (_continuousGeneration)
        {
            _continuousGenerationCoroutine = StartCoroutine(ContinuousGenerationRoutine());
        }
    }
    
    // 定期更新绘制的纹理
    private IEnumerator UpdatePaintTextureRoutine()
    {
        while (true)
        {
            if (_paintableTexture != null && _paintableTexture.Current != null)
            {
                UpdatePaintTexture();
            }
            yield return new WaitForSeconds(0.1f); // 每0.1秒更新一次
        }
    }

    void Update()
    {
        // 检查连续生成状态是否发生变化
        if (_continuousGeneration != _previousContinuousGeneration)
        {
            if (_continuousGeneration)
            {
                // 开始连续生成
                if (_continuousGenerationCoroutine == null)
                {
                    _continuousGenerationCoroutine = StartCoroutine(ContinuousGenerationRoutine());
                }
            }
            else
            {
                // 停止连续生成
                if (_continuousGenerationCoroutine != null)
                {
                    StopCoroutine(_continuousGenerationCoroutine);
                    _continuousGenerationCoroutine = null;
                }
            }
            
            // 更新状态记录
            _previousContinuousGeneration = _continuousGeneration;
        }
        
        if (_stream.isRunning())
        {
            _startButton.GetComponentInChildren<Text>().text = "Pipeline Loaded";
            if (_lastRunningStatus != _stream.isRunning()) UpdateStreamDiff();
        }
        else if (_stream.isPending())
            _startButton.GetComponentInChildren<Text>().text = "Waiting...";
        else
        {
            _startButton.interactable = _stream.isValid();
            _startButton.GetComponentInChildren<Text>().text = "Start StreamDiff";
        }
        _lastRunningStatus = _stream.isRunning();
    }

    void OnDestroy()
    {
        if (_webcamTexture != null) _webcamTexture.Stop();
        _inputMaterial.mainTexture = _originalInputTexture;
        
        if (_paintTexture != null)
        {
            Destroy(_paintTexture);
        }
        
        // 停止连续生成协程
        if (_continuousGenerationCoroutine != null)
        {
            StopCoroutine(_continuousGenerationCoroutine);
            _continuousGenerationCoroutine = null;
        }
    }

    private void ProcessImage(RenderTexture sourceTexture)
    {
        if (sourceTexture == null || !_stream.isRunning())
            return;

        // 初始化图像字节数组
        int width = sourceTexture.width;
        int height = sourceTexture.height;
        int dataLength = width * height * 3; // RGB格式

        if (_imageBytes == null || _imageBytes.Length != dataLength)
        {
            _imageBytes = new byte[dataLength];
        }

        // 创建临时纹理并复制源纹理内容
        RenderTexture.active = sourceTexture;
        Texture2D tempTexture = new Texture2D(width, height, TextureFormat.RGB24, false, QualitySettings.activeColorSpace == ColorSpace.Linear);
        tempTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tempTexture.Apply();

        // 获取像素颜色并转换为RGB字节数组
        Color[] pixels = tempTexture.GetPixels();
        int byteIndex = 0;

        // 像素处理：Unity是RGBA格式，我们需要转为RGB并正确处理颜色空间
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = y * width + x;
                Color color = pixels[pixelIndex];
                
                // 颜色空间处理
                if (QualitySettings.activeColorSpace == ColorSpace.Linear)
                {
                    // 确保颜色从线性空间正确转换到sRGB
                    color = color.linear;
                }

                // 转换为RGB字节
                _imageBytes[byteIndex++] = (byte)Mathf.Clamp(color.r * 255, 0, 255);
                _imageBytes[byteIndex++] = (byte)Mathf.Clamp(color.g * 255, 0, 255);
                _imageBytes[byteIndex++] = (byte)Mathf.Clamp(color.b * 255, 0, 255);
            }
        }

        // 发送到服务端处理
        _stream.InputImage(_imageBytes, width, height);

        // 清理临时纹理
        Destroy(tempTexture);
    }

    // 新增方法：切换绕过模式
    public void ToggleBypassMode(bool isOn)
    {
        _bypassMode = isOn;
        
        if (_stream != null)
        {
            _stream._bypassMode = _bypassMode;
            Debug.Log($"绕过模式已{(_bypassMode ? "启用" : "禁用")}");
            
            // 如果已经加载了Pipeline，则重新加载以应用新设置
            if (_stream.isRunning())
            {
                _startButton.interactable = false;
                _stream.LoadPipeline();
            }
        }
    }
}
