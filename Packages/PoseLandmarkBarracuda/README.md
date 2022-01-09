# PoseLandmarkBarracuda
![demo](https://user-images.githubusercontent.com/34697515/126494250-a7f2520c-886b-46ab-ad69-645c0d7e91da.png)

PoseLandmarkBarracuda is a human pose landmark detecter that runs the [Mediapipe Pose](https://google.github.io/mediapipe/solutions/pose) Landmark neural network model on the [Unity Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@latest).

PoseLandmarkBarracuda implementation is inspired by [HandLandmarkBarracuda](https://github.com/keijiro/HandLandmarkBarracuda) and I referenced [his](https://github.com/keijiro) source code.(Thanks, [keijiro](https://github.com/keijiro)!).

## Install
PoseLandmarkBarracuda can be installed with npm or GitHub URL.

### Install from npm (Recommend)
PoseLandmarkBarracuda can be installed by adding following sections to the manifest file (`Packages/manifest.json`).

To the `scopedRegistries` section:
```
{
  "name": "creativeikep",
  "url": "https://registry.npmjs.com",
  "scopes": [ "jp.ikep" ]
}
```
To the `dependencies` section:
```
"jp.ikep.mediapipe.poselandmark": "1.1.1"
```
Finally, the manifest file looks like below:
```
{
    "scopedRegistries": [
        {
            "name": "creativeikep",
            "url": "https://registry.npmjs.com",
            "scopes": [ "jp.ikep" ]
        }
    ],
    "dependencies": {
        "jp.ikep.mediapipe.poselandmark": "1.1.1",
        ...
    }
}
```

### Install from GitHub URL
PoseLandmarkBarracuda can be installed by adding below URL on the Unity Package Manager's window
```
https://github.com/creativeIKEP/PoseLandmarkBarracuda.git?path=Packages/PoseLandmarkBarracuda#v1.1.1
```
or, adding below sentence to your manifest file(`Packages/manifest.json`) `dependencies` block.
```
"jp.ikep.mediapipe.poselandmark": "https://github.com/creativeIKEP/PoseLandmarkBarracuda.git?path=Packages/PoseLandmarkBarracuda#v1.1.1"
```

## Demo Image
Demo image was downloaded from [here](https://unsplash.com/photos/72zsd_fnxYc).

## ONNX Model
The ONNX model files have been converted for Unity Barracuda from Mediapipe's ["pose_landmark_full.tflite"](https://github.com/google/mediapipe/blob/v0.8.6/mediapipe/modules/pose_landmark/pose_landmark_full.tflite) and ["pose_landmark_lite.tflite"](https://github.com/google/mediapipe/blob/v0.8.6/mediapipe/modules/pose_landmark/pose_landmark_lite.tflite) file.
The ONNX model files were converted with [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow) and [tf2onnx](https://github.com/onnx/tensorflow-onnx).

## Author
[IKEP](https://ikep.jp)

## LICENSE
Copyright (c) 2021 IKEP

[Apache-2.0](/LICENSE.md)
