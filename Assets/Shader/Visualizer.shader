Shader "Hidden/PoseLandmark/Visualizer"
{
    CGINCLUDE

    #include "UnityCG.cginc"

    uint _upperBodyOnly;
    StructuredBuffer<float4> _vertices;

    float4 Vertex(uint vid : SV_VertexID, uint iid : SV_InstanceID): SV_POSITION
    {
        float4 p = _vertices[iid];

        const float size = 0.015;

        float x = p.x + size * lerp(-1, 1, vid == 1) * (vid < 2);
        float y = p.y + size * lerp(-1, 1, vid == 3) * (vid >= 2);
        x = (2 * x - 1) * _ScreenParams.y / _ScreenParams.x;
        y =  2 * y - 1;

        return float4(x, y, 0, 1);
    }

    float4 Fragment(float4 pos: SV_POSITION): SV_Target
    {
        return float4(1, 0, 0, 1);
    }

    ENDCG

    SubShader
    {
        ZWrite Off ZTest Always Cull Off
        Blend SrcAlpha OneMinusSrcAlpha
        Pass
        {
            CGPROGRAM
            #pragma vertex Vertex
            #pragma fragment Fragment
            ENDCG
        }
    }
}