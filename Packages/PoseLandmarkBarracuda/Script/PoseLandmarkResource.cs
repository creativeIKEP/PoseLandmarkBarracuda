using UnityEngine;
using Unity.Barracuda;

namespace Mediapipe.PoseLandmark{
    [CreateAssetMenu(fileName = "PoseLandmark", menuName = "ScriptableObjects/Pose Landmark Resource")]
    public class PoseLandmarkResource : ScriptableObject
    {
        public ComputeShader preProcessCS;
        public ComputeShader postProcessCS;
        public NNModel liteModel;
        public NNModel fullModel;
    }
}