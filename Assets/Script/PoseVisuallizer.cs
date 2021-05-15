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
        landmarker = new PoseLandmarker(poseLandmarkResource);
    }

    void LateUpdate(){
        // Predict pose detection by neural network model.
        landmarker.ProcessImage(webCamInput.inputImageTexture);
    } 

    void OnRenderObject(){
        material.SetInt("_upperBodyOnly", (isUpperBodyOnly ? 1 : 0));
        material.SetBuffer("_vertices", landmarker.outputBuffer);

        material.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Lines, 4, 25);
    }

    void OnApplicationQuit(){
        landmarker.Dispose();
    }
}
