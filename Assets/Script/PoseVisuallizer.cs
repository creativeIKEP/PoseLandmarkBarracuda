using UnityEngine;
using Mediapipe.PoseLandmark;

public class PoseVisuallizer : MonoBehaviour
{
    [SerializeField] WebCamInput webCamInput;
    [SerializeField] Shader shader;
    [SerializeField] PoseLandmarkResource poseLandmarkResource;
    [SerializeField] bool isUpperBodyOnly;

    Material material;
    PoseLandmarker landmarker;


    void Start(){
        material = new Material(shader);
        landmarker = new PoseLandmarker(poseLandmarkResource, isUpperBodyOnly);
    }

    void LateUpdate(){
        // Predict pose landmark by neural network model.
        landmarker.ProcessImage(webCamInput.inputImageTexture, isUpperBodyOnly);
    } 

    void OnRenderObject(){
        material.SetPass(0);
        // Set predicted pose landmark results.
        material.SetBuffer("_vertices", landmarker.outputBuffer);
        // Draw (25 or 33) landmark points.
        Graphics.DrawProceduralNow(MeshTopology.Lines, 4, landmarker.vertexCount);
    }

    void OnApplicationQuit(){
        landmarker.Dispose();
    }
}
