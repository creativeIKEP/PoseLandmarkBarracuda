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
            0~(24 or 32) index datas are pose landmark.
            Check below Mediapipe document about relation between index and landmark position.
            https://google.github.io/mediapipe/solutions/pose#pose_landmarks

            Each data factors are
            x: x cordinate value of pose landmark ([0, 1]).
            y: y cordinate value of pose landmark ([0, 1]).
            z: Landmark depth with the depth at the midpoint of hips being the origin.
               The smaller the value the closer the landmark is to the camera. ([0, 1]).
               This value is full body mode only.
            w: The score of whether the landmark position is visible ([0, 1]).

            (25 or 33) index data is the score whether human pose is visible ([0, 1]).
            This data is (score, 0, 0, 0).
        */
        public ComputeBuffer outputBuffer;
        // Pose segmentation.
        public RenderTexture segmentationRT;
        // Pose landmark point counts.
        public int vertexCount{
            get{
                if(isUpperBodyOnly) return UPPER_BODY_VERTEX_COUNT;
                return FULL_BODY_VERTEX_COUNT;
            }
        }
        #endregion
        
        #region constant number
        // Input image size defined by pose landmark network model.
        const int IMAGE_SIZE = 256;
        // Pose landmark point counts when full body mode.
        // Defined by full body neural network model.
        const int FULL_BODY_VERTEX_COUNT = 33;
        // Pose landmark point counts when upper body mode.
        // Defined by upper body neural network model.
        const int UPPER_BODY_VERTEX_COUNT = 25;
        // Output vector length of full body neural network model. 
        const int FULL_BODY_LD_LEN = 195;
        // Output vector length of upper body neural network model. 
        const int UPPDER_BODY_LD_LEN = 155;
        #endregion

        #region private variable
        // Mode flag which full body or upper body.
        bool isUpperBodyOnly;
        ComputeShader preProcessCS;
        ComputeShader postProcessCS;
        ComputeBuffer networkInputBuffer;
        NNModel fullBodyModel;
        NNModel upperBodyModel;
        Model model;
        IWorker woker;
        #endregion
        
        #region public method
        public PoseLandmarker(PoseLandmarkResource resource, bool isUpperBody){
            preProcessCS = resource.preProcessCS;
            postProcessCS = resource.postProcessCS;
            fullBodyModel = resource.fullBodyModel;
            upperBodyModel = resource.upperBodyModel;

            networkInputBuffer = new ComputeBuffer(IMAGE_SIZE * IMAGE_SIZE * 3, sizeof(float));
            segmentationRT = new RenderTexture(128, 128, 0, RenderTextureFormat.ARGB32);

            // Initialize related with mode which full body or upper body.
            ExchangeModel(isUpperBody);
        }

        public void ProcessImage(Texture inputTexture, bool isUpperBody){
            if(isUpperBodyOnly != isUpperBody){
                // Reinitialize variables related with mode which full body or upper body
                // if mode of this frame was changed from previous mode.
                ExchangeModel(isUpperBody);
            }

            // Resize `inputTexture` texture to network model image size.
            preProcessCS.SetTexture(0, "_inputTexture", inputTexture);
            preProcessCS.SetBuffer(0, "_output", networkInputBuffer);
            preProcessCS.Dispatch(0, IMAGE_SIZE / 8, IMAGE_SIZE / 8, 1);

            //Execute neural network model.
            var inputTensor = new Tensor(1, IMAGE_SIZE, IMAGE_SIZE, 3, networkInputBuffer);
            woker.Execute(inputTensor);
            inputTensor.Dispose();

            // Convert 4 dimensions Tensor to 1 dimension ComputeBuffer.
            var poseFlagBuffer = TensorToBuffer("output_poseflag", 1);
            var landmarkBuffer = TensorToBuffer("ld_3d", (isUpperBodyOnly ? UPPDER_BODY_LD_LEN : FULL_BODY_LD_LEN));
            
            // Get final results of pose landmark.
            postProcessCS.SetInt("_keypointCount", vertexCount);
            postProcessCS.SetBuffer(0, "_poseFlag", poseFlagBuffer);
            postProcessCS.SetBuffer(0, "_Landmark", landmarkBuffer);
            postProcessCS.SetBuffer(0, "_Output", outputBuffer);
            postProcessCS.Dispatch(0, 1, 1, 1);

            // Set pose landmark segmentation texture.
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
        #endregion

        #region private method
        // Reinitialize variables related with mode which full body or upper body.
        void ExchangeModel(bool isUpperBody){
            outputBuffer?.Dispose();
            woker?.Dispose();

            // Reinitialize 'outputBuffer' with new buffer length.
            var vertexCount = isUpperBody ? UPPER_BODY_VERTEX_COUNT : FULL_BODY_VERTEX_COUNT;
            outputBuffer = new ComputeBuffer(vertexCount + 1, sizeof(float) * 4);

            // Switch neural network models between full body and upper body.
            NNModel nnModel = isUpperBody ? upperBodyModel : fullBodyModel;
            model = ModelLoader.Load(nnModel);
            woker = model.CreateWorker();

            // Switch control flag.
            isUpperBodyOnly = isUpperBody;
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
