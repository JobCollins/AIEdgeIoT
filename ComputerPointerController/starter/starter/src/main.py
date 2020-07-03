from face_detection import Face_detection
from facial_landmarks_detection import Landmark_Detection
from head_pose_estimation import Head_pose
from gaze_estimation import Gaze_estimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

import cv2
import numpy as np
import time
import os
from argparse import ArgumentParser


def build_argparser():

    """
    Parse command line arguments.
    :return: command line arguments
    """

    parser = ArgumentParser()
    parser.add_argument("-fd", "--faceDetectionModel", required=True, type=str, 
                        help="Path to an xml file with a trained model.")

    parser.add_argument("-fl", "--facialLandmarksModel", required=True, type=str,
                        help="Path to an xml file with a trained model.")

    parser.add_argument("-hp", "--headPoseEstimationModel", required=True, type=str, 
                        help="Path to an xml file with a trained model.")

    parser.add_argument("-gz", "--gazeEstimationModel", required=True, type=str, 
                        help="Path to an xml file with a trained model.")

    parser.add_argument("-i", "--input", required=True, type=str, 
                    help="Path to webcam or video file")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                    help="Probability threshold for detections filtering")

    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="Specify the target device to infer on:"
                             "CPU, GPU, FPGA or MYRIAD is acceptable")

    parser.add_argument("-o", '--output_path', default='/results/', type=str)

    return parser
    
    