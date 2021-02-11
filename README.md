# AIEdgeIoT

Repository of my Intel Edge AI for IoT Nanodegree projects

## People Counter App

This app uses a person detection TensorFlow model(ssd_mobilenet_v2_coco) from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md. To download vist the link then,

1. Select the `ssd_mobilenet_v2_coco` model
2. A `tar.gz` file is downloaded
3. Un-tar the file using the command `tar-xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

Within the untar'ed `ssd_mobilenet_v2_coco_2018_03_29` folder you will find the model files.
To convert the model into an Intermediate Representation for use with the Model Optimizer I used the command

`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
`

Utilizing the Inference Engine, I use the model to perform inference on an input video, and extract useful data concerning the count of people in frame and how long they stay in frame. The app then sends this information over MQTT, as well as sending the output frame, in order to view it from a separate UI server over a network.


## Smart Queueing System

This project uses the Intel Dev Cloud to build and test AI at the edge. Specifically, the app reduces congestion in queueing scenario. The app uses the Intel OpenVINO API and the person detection model https://docs.openvinotoolkit.org/2019_R1/_person_detection_retail_0013_description_person_detection_retail_0013.html) from the Intel Open Model Zoo to count people in a queue so as to direct them to the least congested queue.

The three scenarios you'll be looking at are:

1. Scenario 1: Manufacturing Sector
2. Scenario 2: Retail Sector
3. Scenario 3: Transportation Sector

### Hardware Proposal

All of the scenarios involve people in queues, but each scenario will require different hardware. So the first task will be to determine which hardware might work for each scenario—and then explain the initial choice in a proposal document.

Later, after building and testing the application on each hardware device for all three scenarios, a review of the results is done and validate or update the initial proposed choices in the proposal document.

### Testing the Hardware

With the initial hypothesis about what hardware might work for the client, it's time to test it and see how it performs!

In the SmartQueueingSystem/ProjectThree folder, you'll find notebooks where there's a build out of the smart queuing application and test of its performance on all four different hardware types (CPU, IGPU, VPU, and FPGA) using the DevCloud.


# Computer Pointer Controller

The Computer Pointer Controller is an AI project that is part of the Intel Edge AI for IoT Nanodegree curriculum. In this project, the ability to run multiple models in the same machine and coordinate the flow of data between those models.

The project uses the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. Along with this model, the face detection, landmarks regression, and head pose estimation models are used.

***Check the ComputerPointerController/starter/starter folder to view the project.***
