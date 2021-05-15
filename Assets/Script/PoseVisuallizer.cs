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
        // Predict pose detection by neural network model.
        landmarker.ProcessImage(webCamInput.inputImageTexture, isUpperBodyOnly);
    } 

    void OnRenderObject(){
        material.SetPass(0);
        material.SetBuffer("_vertices", landmarker.outputBuffer);
        Graphics.DrawProceduralNow(MeshTopology.Lines, 4, (isUpperBodyOnly ? 25 : 33));
    }

    void OnApplicationQuit(){
        landmarker.Dispose();
    }
}
