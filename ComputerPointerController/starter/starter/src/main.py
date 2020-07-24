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
import logging


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

    parser.add_argument("-o", '--output_path', default='./results/', type=str)

    parser.add_argument("-flag", "--bbox_flag", required=False, nargs='+',
                        default=[],
                        help="flag ffd ffl fhp fgz (Seperated by space)"
                             "for model outputs detections of each frame,"
                             "ffd for Face Detection Model, ffl for Facial Landmark Detection Model"
                             "fhp for Head Pose Estimation Model, fgz for Gaze Estimation Model.")  #changed CLI command

    return parser


def draw_bbox(frame, bbox_flag, image_copy, l_eye, r_eye, face_bbox, eyes, hp_output, gaze_coords):

    bbox_frame = frame.copy()

    if 'ffd' in bbox_flag:
        if len(bbox_flag) != 1:
            bbox_frame = image_copy
        cv2.rectangle(frame, (face_bbox[0][0], face_bbox[0][1]), (face_bbox[0][2], face_bbox[0][3]), (0, 0, 0), 3)

    if 'ffl' in bbox_flag:
        cv2.rectangle(image_copy, (eyes[0][0], eyes[0][1]), (eyes[0][2], eyes[0][3]), (255, 0, 0), 2)
        cv2.rectangle(image_copy, (eyes[1][0], eyes[1][1]), (eyes[1][2], eyes[1][3]), (255, 0, 0), 2)

    if 'fhp' in bbox_flag:   #changed from fh to fhp
        cv2.putText(
            frame,
            "Head Pose Angles: yaw= {:.2f} , pitch= {:.1f} , roll= {:.1f}".format(
                hp_output[0], hp_output[1], hp_output[2]), (20, 20),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    if 'fgz' in bbox_flag:   #changed from fg to fgz

        cv2.putText(
            frame,
            "Gaze Coords: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                gaze_coords[0], gaze_coords[1], gaze_coords[2]), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)


    return bbox_frame

def main():

    args = build_argparser().parse_args()
    logger = logging.getLogger('main')

    model_path_dict = {
        'FaceDetectionModel': args.faceDetectionModel,
        'FacialLandmarksModel': args.facialLandmarksModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel
    }
    
    bbox_flag = args.bbox_flag
    input_filename = args.input
    device_name = args.device
    prob_threshold = args.prob_threshold
    output_path = args.output_path

    if input_filename.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_filename):
            logger.error("Unable to find specified video file")
            exit(1)
        feeder = InputFeeder(input_type='video', input_file=input_filename)

    for model_path in list(model_path_dict.values()):
        if not os.path.isfile(model_path):
            logger.error("Unable to find specified model file" + str(model_path))
            exit(1)

    face_detection_model= Face_detection(model_path_dict['FaceDetectionModel'], device_name, threshold=prob_threshold)
    facial_landmarks_detection_model = Landmark_Detection(model_path_dict['FacialLandmarksModel'], device_name, threshold=prob_threshold)
    head_pose_estimation_model = Head_pose(model_path_dict['HeadPoseEstimationModel'], device_name, threshold=prob_threshold)
    gaze_estimation_model = Gaze_estimation(model_path_dict['GazeEstimationModel'], device_name, threshold=prob_threshold)

    is_benchmarking = False

    if not is_benchmarking:
        mouse_controller = MouseController('medium', 'fast')

    start_model_load_time = time.time()
    face_detection_model.load_model()
    facial_landmarks_detection_model.load_model()
    head_pose_estimation_model.load_model()
    gaze_estimation_model.load_model()
    total_model_load_time = time.time() - start_model_load_time

    feeder.load_data()

    out_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), int(feeder.get_fps()/10),
                                (1920, 1080), True)

    frame_count = 0
    start_inference_time = time.time()
    for ret, frame in feeder.next_batch():

        if not ret:
            break

        frame_count += 1

        key = cv2.waitKey(60)

        try:
            face_bbox, image_copy = face_detection_model.predict(frame)

            if type(image_copy) == int:
                logger.warning("Unable to detect the face")
                if key == 27:
                    break
                continue

            l_eye, r_eye, eyes = facial_landmarks_detection_model.predict(image_copy)
            hp_output = head_pose_estimation_model.predict(image_copy)
            mouse_coords, gaze_coords = gaze_estimation_model.predict(l_eye, r_eye, hp_output)

        except Exception as e:
            logger.warning("Could predict using model" + str(e) + " for frame " + str(frame_count))
            continue

        image = cv2.resize(frame, (500, 500))

        if not len(bbox_flag) == 0:
            bbox_frame = draw_bbox(
                frame, bbox_flag, image_copy, l_eye, r_eye,
                face_bbox, eyes, hp_output, gaze_coords)
            image = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(bbox_frame, (500, 500))))

        cv2.imshow('preview', image)
        out_video.write(frame)

        if frame_count % 5 == 0 and not is_benchmarking:
            mouse_controller.move(mouse_coords[0], mouse_coords[1])

        if key == 27:
            break

    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time

    try:
        os.mkdir(output_path)
    except OSError as error:
        logger.error(error)

    with open(output_path+'stats.txt', 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(total_model_load_time) + '\n')

    logger.info('Model load time: ' + str(total_model_load_time))
    logger.info('Inference time: ' + str(total_inference_time))
    logger.info('FPS: ' + str(fps))

    logger.info('Video stream ended')
    cv2.destroyAllWindows()
    feeder.close()


if __name__ == '__main__':
    main()