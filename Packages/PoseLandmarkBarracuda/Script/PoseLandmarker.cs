using UnityEngine;
using Unity.Barracuda;

namespace Mediapipe.PoseLandmark
{
    public class PoseLandmarker : System.IDisposable
    {
        #region public variables
        /*
        Pose landmark result Buffer.
            'outputBuffer' is array of float4 type.
            0~32 index datas are pose landmark.
            Check below Mediapipe document about relation between index and landmark position.
            https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazepose-ghum-3d

            Each data factors are
            x: x cordinate value of pose landmark ([0, 1]).
            y: y cordinate value of pose landmark ([0, 1]).
            z: Landmark depth with the depth at the midpoint of hips being the origin.
               The smaller the value the closer the landmark is to the camera. ([0, 1]).
               This value is full body mode only.
            w: The score of whether the landmark position is visible ([0, 1]).

            33 index data is the score whether human pose is visible ([0, 1]).
            This data is (score, 0, 0, 0).
        */
        public ComputeBuffer outputBuffer;
        /*
        Pose world landmark result Buffer.
            'worldLandmarkBuffer' is array of float4 type.
            0~32 index datas are pose world landmark.

            Each data factors are
            x, y and z: Real-world 3D coordinates in meters with the origin at the center between hips.
            w: The score of whether the world landmark position is visible ([0, 1]).

            33 index data is the score whether human pose is visible ([0, 1]).
            This data is (score, 0, 0, 0).
        */
        public ComputeBuffer worldLandmarkBuffer;
        // Pose segmentation.
        public RenderTexture segmentationRT;
        // Pose landmark point counts.
        public int vertexCount => BODY_VERTEX_COUNT;
        #endregion
        
        #region constant number
        // Input image size defined by pose landmark network model.
        const int IMAGE_SIZE = 256;
        // Pose landmark point counts.
        // Defined by full body neural network model.
        const int BODY_VERTEX_COUNT = 33;
        // Output vector length of network model. 
        const int BODY_LD_LEN = 195;
        // World landmark output vector length of network model. 
        const int WORLD_LD_LEN = 117;
        // Pose segmentation texture size.
        const int SEGMENTATION_SIZE = 128;
        #endregion

        #region private variable
        ComputeShader preProcessCS;
        ComputeShader postProcessCS;
        ComputeBuffer networkInputBuffer;
        NNModel liteModel;
        NNModel fullModel;
        Model model;
        IWorker woker;
        PoseLandmarkModel selectedModel;
        #endregion
        
        #region public method
        public PoseLandmarker(PoseLandmarkResource resource, PoseLandmarkModel poseLandmarkModel = PoseLandmarkModel.full){
            preProcessCS = resource.preProcessCS;
            postProcessCS = resource.postProcessCS;
            liteModel = resource.liteModel;
            fullModel = resource.fullModel;

            networkInputBuffer = new ComputeBuffer(IMAGE_SIZE * IMAGE_SIZE * 3, sizeof(float));
            segmentationRT = new RenderTexture(SEGMENTATION_SIZE, SEGMENTATION_SIZE, 0, RenderTextureFormat.ARGB32);
            outputBuffer = new ComputeBuffer(vertexCount + 1, sizeof(float) * 4);
            worldLandmarkBuffer = new ComputeBuffer(vertexCount + 1, sizeof(float) * 4);

            // Initialize related with mode which full body or upper body.
            ExchangeModel(poseLandmarkModel);
        }

        public void ProcessImage(Texture inputTexture, PoseLandmarkModel poseLandmarkModel = PoseLandmarkModel.full){
            // Resize `inputTexture` texture to network model image size.
            preProcessCS.SetTexture(0, "_inputTexture", inputTexture);
            preProcessCS.SetBuffer(0, "_output", networkInputBuffer);
            preProcessCS.Dispatch(0, IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);

            ProcessImage(networkInputBuffer, poseLandmarkModel);
        }

        public void ProcessImage(ComputeBuffer input, PoseLandmarkModel poseLandmarkModel = PoseLandmarkModel.full){
            if(selectedModel != poseLandmarkModel){
                // Reinitialize variables related with modes if mode of this frame was changed from previous mode.
                ExchangeModel(poseLandmarkModel);
            }

            //Execute neural network model.
            var inputTensor = new Tensor(1, IMAGE_SIZE, IMAGE_SIZE, 3, input);
            woker.Execute(inputTensor);
            inputTensor.Dispose();

            // Convert 4 dimensions Tensor to 1 dimension ComputeBuffer.
            var poseFlagBuffer = TensorToBuffer("Identity_1", 1);
            var landmarkBuffer = TensorToBuffer("Identity", BODY_LD_LEN);
            var worldLandmarkRawBuffer = TensorToBuffer("Identity_4", WORLD_LD_LEN);
            
            // Get final results of pose landmark.
            postProcessCS.SetInt("_keypointCount", vertexCount);
            postProcessCS.SetBuffer(0, "_poseFlag", poseFlagBuffer);
            postProcessCS.SetBuffer(0, "_Landmark", landmarkBuffer);
            postProcessCS.SetBuffer(0, "_LandmarkWorld", worldLandmarkRawBuffer);
            postProcessCS.SetBuffer(0, "_Output", outputBuffer);
            postProcessCS.SetBuffer(0, "_OutputWorld", worldLandmarkBuffer);
            postProcessCS.Dispatch(0, 1, 1, 1);

            // Set pose landmark segmentation texture.
            var segTemp = CopyOutputToTempRT("Identity_2", SEGMENTATION_SIZE, SEGMENTATION_SIZE);
            Graphics.Blit(segTemp, segmentationRT);
            RenderTexture.ReleaseTemporary(segTemp);
        }

        public void Dispose(){
            networkInputBuffer?.Dispose();
            outputBuffer?.Dispose();
            worldLandmarkBuffer?.Dispose();
            segmentationRT.Release();
            woker?.Dispose();
        }
        #endregion

        #region private method
        // Reinitialize variables related with modes.
        void ExchangeModel(PoseLandmarkModel poseLandmarkModel){
            woker?.Dispose();

            // Switch neural network models.
            NNModel nnModel;
            switch(poseLandmarkModel){
                case PoseLandmarkModel.lite:
                    nnModel = liteModel;
                    break;
                case PoseLandmarkModel.full:
                    nnModel = fullModel;
                    break;
                default:
                    nnModel = fullModel;
                    break;
            }
            model = ModelLoader.Load(nnModel);
            woker = model.CreateWorker();

            // Switch control flag.
            selectedModel = poseLandmarkModel;
        }
        
        // Extract the vector in the 4 dimensions Tensor as a Compute Buffer.
        ComputeBuffer TensorToBuffer(string name, int length){
            var shape = new TensorShape(length);
            var tensor = woker.PeekOutput(name).Reshape(shape);
            var buffer = ((ComputeTensorData)tensor.data).buffer;
            tensor.Dispose();
            return buffer;
        }

        // Exchange network output tensor to RenderTexture.
        RenderTexture CopyOutputToTempRT(string name, int w, int h)
        {
            var rtFormat = RenderTextureFormat.ARGB32;
            var shape = new TensorShape(1, h, w, 1);
            var rt = RenderTexture.GetTemporary(w, h, 0, rtFormat);
            var tensor = woker.PeekOutput(name).Reshape(shape);
            tensor.ToRenderTexture(rt);
            tensor.Dispose();
            return rt;
        }
        #endregion
    }
}
