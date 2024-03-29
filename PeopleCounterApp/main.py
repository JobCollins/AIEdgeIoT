"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def observe(frame, result):
    current_count = 0
    for item in result[0][0]:
        # Draw bounding box for object
        if item[2] > prob_threshold:
            xmin = int(item[3] * initial_w)
            ymin = int(item[4] * initial_h)
            xmax = int(item[5] * initial_w)
            ymax = int(item[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
        
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    client = connect_mqtt()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    args = build_argparser().parse_args()
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    n,c,h,w = infer_network.load_model(args.model, args.device, args.cpu_extension)[1]
    
    ### TODO: Handle the input stream ###
    image_flag=False
    if args.input == 'CAM':
        args.input=0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag=True
        
    video_cap = cv2.VideoCapture(args.input)
    video_cap.open(args.input)
    
    global initial_w, initial_h
    initial_w = video_cap.get(3)
    initial_h = video_cap.get(4)

    last_count = 0
    total_count = 0
    start_time = 0

    ### TODO: Loop until stream is over ###
    counter = 0
    while video_cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = video_cap.read()
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        counter +=1
        
        
        ### TODO: Pre-process the image as needed ###
        #resize frame
        image = cv2.resize(frame, (w,h))
        image = image.transpose((2,0,1))
        image = image.reshape((n, c, h, w))
        
        ### TODO: Start asynchronous inference for specified request ###
        request_id = 0
        infer_start = time.time()
        infer_network.exec_net(request_id, image)
        
             
            
        
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id)==0:
            
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            ### TODO: Extract any desired stats from the results ###
            
            frame, current_count = observe(frame, result)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###                
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if counter % 35 == 0:
                if current_count > last_count:
                    entry_time = time.time()
                    total_count = total_count + current_count - last_count
                    client.publish("person", json.dumps({"total": total_count}))

                elif current_count < last_count:
                    time_taken = int(time.time() - entry_time)
                    client.publish("person/duration", json.dumps({"duration": time_taken}))

                client.publish("person", json.dumps({"count": current_count}))
                last_count = current_count

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
            
    video_cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
