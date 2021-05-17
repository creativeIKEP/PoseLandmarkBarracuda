# PoseLandmarkBarracuda
![demo](/screenshot/demo.png)

PoseLandmarkBarracuda is a human pose landmark detecter that runs the [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose) Landmark neural network model on the [Unity Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@latest).

PoseLandmarkBarracuda implementation is inspired by [HandLandmarkBarracuda](https://github.com/keijiro/HandLandmarkBarracuda) and I referenced [his](https://github.com/keijiro) source code.(Thanks, [keijiro](https://github.com/keijiro)!).

### Install
PoseLandmarkBarracuda can be installed by adding below URL on the Unity Package Manager's window.
```
https://github.com/creativeIKEP/PoseLandmarkBarracuda.git?path=Packages/PoseLandmarkBarracuda
```

### Demo Image
Demo image was downloaded from [here](https://unsplash.com/photos/72zsd_fnxYc).

### ONNX Model
The ONNX model files have been converted for Unity Barracuda from Mediapipe's ["pose_landmark_full_body.tflite"](https://github.com/google/mediapipe/blob/0.8.3.2/mediapipe/modules/pose_landmark/pose_landmark_full_body.tflite) and ["pose_landmark_upper_body.tflite"](https://github.com/google/mediapipe/blob/0.8.3.2/mediapipe/modules/pose_landmark/pose_landmark_upper_body.tflite) file.
The conversion operation is the same as [FaceLandmarkBarracuda](https://github.com/keijiro/FaceLandmarkBarracuda) by [keijiro](https://github.com/keijiro).
Check [his operation script](https://colab.research.google.com/drive/1C6zEB3__gcHEWnWRm-b4jIA0srA1gkyq?usp=sharing) for details.

### Author
[IKEP](https://ikep.jp)

### LICENSE
Copyright (c) 2021 IKEP

[Apache-2.0](/LICENSE.md)
