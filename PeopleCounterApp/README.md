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

