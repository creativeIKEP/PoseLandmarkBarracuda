using UnityEngine;
using UnityEngine.UI;
using Mediapipe.PoseLandmark;

public class PoseVisuallizer : MonoBehaviour
{
    [SerializeField] WebCamInput webCamInput;
    [SerializeField] Shader shader;
    [SerializeField] RawImage inputImageUI;
    [SerializeField] RawImage segmentationImage;
    [SerializeField] PoseLandmarkResource poseLandmarkResource;
    [SerializeField] PoseLandmarkModel poseLandmarkModel;

    Material material;
    PoseLandmarker landmarker;


    void Start(){
        material = new Material(shader);
        landmarker = new PoseLandmarker(poseLandmarkResource, poseLandmarkModel);
    }

    void LateUpdate(){
        inputImageUI.texture = webCamInput.inputImageTexture;

        // Predict pose landmark by neural network model.
        landmarker.ProcessImage(webCamInput.inputImageTexture, poseLandmarkModel);
    } 

    void OnRenderObject(){
        segmentationImage.texture = landmarker.segmentationRT;

        var w = inputImageUI.rectTransform.rect.width;
        var h = inputImageUI.rectTransform.rect.height;

        material.SetPass(0);
        // Set predicted pose landmark results.
        material.SetBuffer("_vertices", landmarker.outputBuffer);
        material.SetVector("_uiScale", new Vector2(w, h));
        // Draw (25 or 33) landmark points.
        Graphics.DrawProceduralNow(MeshTopology.Lines, 4, landmarker.vertexCount);
    }

    void OnApplicationQuit(){
        landmarker.Dispose();
    }
}
