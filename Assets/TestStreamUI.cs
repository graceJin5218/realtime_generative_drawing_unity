using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TestStreamUI : MonoBehaviour
{
    public StreamDiffusionClient _stream;
    public Material _inputMaterial;
    public Button _startButton;
    public InputField _promptInput;

    private WebCamTexture _webcamTexture = null;
    private Texture2D _inputTexture = null;
    private Texture _originalInputTexture = null;
    private bool _lastRunningStatus = false;

    public void StartWebcam()
    {
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
        if (_stream.isValid() && !_stream.isRunning()) _stream.LoadPipeline();
    }

    public void UpdateStreamDiff()
    {
        if (_stream.isRunning() && !_stream.isGenerating())
            _stream.AdvancePipeline(_inputTexture, _promptInput.text);
    }

    private IEnumerator UpdateWebcamData()
    {
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
        _inputTexture = _inputMaterial.mainTexture as Texture2D;
    }

    void Update()
    {
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
    }
}
