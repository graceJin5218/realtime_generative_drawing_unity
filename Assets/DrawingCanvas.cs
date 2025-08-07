// DrawingCanvas.cs

using UnityEngine;

[RequireComponent(typeof(Camera))]
public class DrawingCanvas : MonoBehaviour
{
    public RenderTexture canvasTexture;
    public Material brushMaterial;
    public float brushSize = 0.05f;
    public Color brushColor = Color.black;

    private Camera _cam;
    private Vector2? _lastDrawUV;

    void Start()
    {
        _cam = GetComponent<Camera>();
        _cam.orthographic = true;
        _cam.clearFlags = CameraClearFlags.SolidColor;
        _cam.backgroundColor = Color.white;
        _cam.targetTexture = canvasTexture;
    }

    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            Debug.Log("Mouse down");

            Ray ray = _cam.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                Vector2 uv = hit.textureCoord;
                DrawAtUV(uv);
                _lastDrawUV = uv;
            }
        }
        else
        {
            _lastDrawUV = null;
        }
    }

    void DrawAtUV(Vector2 uv)
    {
        Debug.Log($"Drawing at UV: {uv}");

        RenderTexture.active = canvasTexture;
        GL.PushMatrix();
        GL.LoadOrtho();

        brushMaterial.SetColor("_Color", brushColor);
        brushMaterial.SetPass(0);

        GL.Begin(GL.QUADS);

        float s = brushSize;
        float x = uv.x;
        float y = uv.y;

        GL.Vertex3(x - s, y - s, 0);
        GL.Vertex3(x + s, y - s, 0);
        GL.Vertex3(x + s, y + s, 0);
        GL.Vertex3(x - s, y + s, 0);

        //GL.TexCoord2(0, 0); GL.Vertex3(x - size, y - size, 0);
        // GL.TexCoord2(1, 0); GL.Vertex3(x + size, y - size, 0);
        // GL.TexCoord2(1, 1); GL.Vertex3(x + size, y + size, 0);
        // GL.TexCoord2(0, 1); GL.Vertex3(x - size, y + size, 0);

        GL.End();
        GL.PopMatrix();

        RenderTexture.active = null;
    }

    public Texture2D GetDrawingTexture()
    {
        RenderTexture.active = canvasTexture;
        Texture2D tex = new Texture2D(canvasTexture.width, canvasTexture.height, TextureFormat.RGBA32, false);
        tex.ReadPixels(new Rect(0, 0, canvasTexture.width, canvasTexture.height), 0, 0);
        tex.Apply();
        RenderTexture.active = null;
        return tex;
    }

    public void ClearCanvas(Color clearColor)
    {
        RenderTexture.active = canvasTexture;
        GL.Clear(true, true, clearColor);
        RenderTexture.active = null;
    }
}
