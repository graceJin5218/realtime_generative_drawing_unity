Shader "Unlit/Color"
{
    Properties{ _Color("Color", Color) = (1,1,1,1) }
        SubShader
    {
        Tags { "RenderType" = "Opaque" }
        Pass
        {
            Color[_Color]
        }
    }
}
