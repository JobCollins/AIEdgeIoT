# Computer Pointer Controller

The Computer Pointer Controller is an AI project that is part of the Intel Edge AI for IoT Nanodegree curriculum. In this project, the ability to run multiple models in the same machine and coordinate the flow of data between those models.

The project uses the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. Along with this model, the face detection, landmarks regression, and head pose estimation models are used.

## Project Set Up and Installation

1. Download and install the Intel OpenVINO toolkit, you can find the installation guide here https://docs.openvinotoolkit.org/latest/index.html
2. Clone the Computer Pointer Controller project.
3. `cd` into the `path\to\starter\starter` directory. 
4. Create a virtual environment `virtualenv mouse`.
5. Run `pip install requirements.txt` to install the app's dependencies.
6. Download  the models above from **Open Model Zoo** by running
    `path\to\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py --name gaze-estimation-adas-0002`

    `path\to\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py --name face-detection-adas-binary-0001`

    `path\to\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py --name head-pose-estimation-adas-0001`

    `path\to\openvino\deployment_tools\open_model_zoo\tools\downloader\downloader.py --name landmarks-regression-retail-0009`
7. `cd` back to the `path\to\starter\starter` directory.

## Demo

1. In the `path\to\starter\starter` directory, activate the `mouse` virtual environment
    `mouse\Scripts\activate`
2. Initialize the OpenVINO environment
    `cd path\to\openvino\bin\setupvars.bat`
3. Go back to the project directory 
    `cd path\to\starter\starter\src`
4. Run the application using the following commands
    `python main.py -fd ..\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml -fl ..\intel\landmarks-regression-retail-0009\FP32-INT8\landmarks-regression-retail-0009.xml -hp ..\intel\head-pose-estimation-adas-0001\FP32-INT8\head-pose-estimation-adas-0001.xml -gz ..\intel\gaze-estimation-adas-0002\FP32-INT8\gaze-estimation-adas-0002.xml -i ..\bin\demo.mp4 -flag ffd ffl fhp fgz`

At this point a small video window should appear and start controlling your mouse depending on the gazes in the video.

![outputVideo](./src/output_video.mp4)

## Documentation

**src** folder

This folder holds the application (files).
1. `model.py` has the parent class `class Model_X` which holds code that is common across the other model files. This helps to prevent DRY.
2. `face_detection.py` holds the code execution of the face detection model.
3. `facial_landmarks_detction.py` holds the code execution of the landmark regression model.
4. `head_pose_estimation.py` holds the code execution of the head pose estimation model.
5. `gaze_estimation.py` holds the code execution of the gaze estimation model.

Loading the model is executed by the `load_model` function.
Preprocessing inputs is done by `preprocess_inputs` function.
Inferencing is made by the `predict` function
Preprocessing outputs is done by `preprocess_outputs`

***Each model has a different execution of the above functions***


**App Pipeline & Flow of Data**

![pipeline](pipeline.png)


**intel** folder
This folder contains the above downloaded (Open Model Zoo) models in their IR format.

**bin** folder
This folder contains the `demo.mp4` video file that is used to test the models & application.



## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
