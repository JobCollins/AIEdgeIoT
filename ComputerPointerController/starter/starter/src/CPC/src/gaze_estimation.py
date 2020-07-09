from model import Model_X
import math

class Gaze_estimation(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model_X.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Gaze Esimation'
        self.input_name = [k for k in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [k for k in self.model.outputs.keys()]

    
    def predict(self, left_eye, right_eye, hp_coords, request_id=0):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)
        self.network.start_async(
            request_id, 
            inputs={
            'left_eye_image': left_eye,
            'right_eye_image': right_eye,
            'head_pose_angles': hp_coords #changed from coords to hp_coords
            })

        if self.wait() == 0:
            outputs = self.network.requests[0].outputs
            mouse_coords, gaze_coords = self.preprocess_output(outputs, hp_coords)

        return mouse_coords, gaze_coords


    def preprocess_output(self, outputs, hp_coords):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        mouse_coords = (0, 0)
        gaze_coords = outputs[self.output_name[0]][0]  #bracket mistakes moved

        angle_r_fc = hp_coords[2]
        sin_r = math.sin(angle_r_fc * math.pi / 180.0)
        cos_r = math.cos(angle_r_fc * math.pi / 180.0)

        x = gaze_coords[0] * cos_r + gaze_coords[1] * sin_r
        y = -gaze_coords[0] * sin_r + gaze_coords[1] * cos_r

        mouse_coords = (x, y)

        return mouse_coords, gaze_coords
        
