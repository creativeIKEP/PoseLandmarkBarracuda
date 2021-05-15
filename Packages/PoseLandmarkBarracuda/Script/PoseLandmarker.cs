using UnityEngine;
using Unity.Barracuda;

namespace Mediapipe.PoseLandmark
{
    public class PoseLandmarker : System.IDisposable
    {
        const int IMAGE_SIZE = 256;
        const int VERTEX_COUNT = 25;

        ComputeShader preProcessCS;
        ComputeShader postProcessCS;
        ComputeBuffer networkInputBuffer;
        public ComputeBuffer outputBuffer;
        public RenderTexture segmentationRT;

        Model model;
        IWorker woker;

        public PoseLandmarker(PoseLandmarkResource resource){
            preProcessCS = resource.preProcessCS;
            postProcessCS = resource.postProcessCS;

            networkInputBuffer = new ComputeBuffer(IMAGE_SIZE * IMAGE_SIZE * 3, sizeof(float));
            outputBuffer = new ComputeBuffer(VERTEX_COUNT + 1, sizeof(float) * 4);
            segmentationRT = new RenderTexture(128, 128, 0, RenderTextureFormat.RFloat);

            model = ModelLoader.Load(resource.model);
            woker = model.CreateWorker();
        }

        public void ProcessImage(Texture inputTexture){
            // Resize `inputTexture` texture to network model image size.
            preProcessCS.SetTexture(0, "_inputTexture", inputTexture);
            preProcessCS.SetBuffer(0, "_output", networkInputBuffer);
            preProcessCS.Dispatch(0, IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);

            //Execute neural network model.
            var inputTensor = new Tensor(1, IMAGE_SIZE, IMAGE_SIZE, 3, networkInputBuffer);
            woker.Execute(inputTensor);
            inputTensor.Dispose();

            var poseFlagBuffer = TensorToBuffer("output_poseflag", 1);
            var landmarkBuffer = TensorToBuffer("ld_3d", 155);
            
            postProcessCS.SetBuffer(0, "_poseFlag", poseFlagBuffer);
            postProcessCS.SetBuffer(0, "_Landmark", landmarkBuffer);
            postProcessCS.SetBuffer(0, "_Output", outputBuffer);
            postProcessCS.Dispatch(0, 1, 1, 1);

            var segTemp = CopyOutputToTempRT("output_segmentation", 128, 128);
            Graphics.Blit(segTemp, segmentationRT);
            RenderTexture.ReleaseTemporary(segTemp);
        }

        public void Dispose(){
            networkInputBuffer.Dispose();
            outputBuffer.Dispose();
            segmentationRT.Release();
            woker.Dispose();
        }
        
        ComputeBuffer TensorToBuffer(string name, int length){
            var shape = new TensorShape(length);
            var tensor = woker.PeekOutput(name).Reshape(shape);
            var buffer = ((ComputeTensorData)tensor.data).buffer;
            tensor.Dispose();
            return buffer;
        }

        RenderTexture CopyOutputToTempRT(string name, int w, int h)
        {
            var rtFormat = RenderTextureFormat.RFloat;
            var shape = new TensorShape(1, h, w, 1);
            var rt = RenderTexture.GetTemporary(w, h, 0, rtFormat);
            var tensor = woker.PeekOutput(name).Reshape(shape);
            tensor.ToRenderTexture(rt);
            tensor.Dispose();
            return rt;
        }
    }
}
