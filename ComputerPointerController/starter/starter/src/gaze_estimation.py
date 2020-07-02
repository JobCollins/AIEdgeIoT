'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model_X.__init__(self, model_name, device='CPU',extensions=None, threshold=0.6)
        self.model_name = 'Gaze Esimation'
        self.input_name = [k for k in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [k for k in self.model.outputs.keys()]

    
    def predict(self, left_eye, right_eye, coords, request_id):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)
        self.network.start_async(
            request_id=0, 
            inputs={
            'left_eye': left_eye,
            'right_eye': right_eye,
            'head_pose_coords': coords
            })

        if self.wait() == 0:
            outputs = self.network.requests[0].outputs
            mouse_coords, gaze_coords = self.preprocess_output(outputs, coords)

        return mouse_coords, gaze_coords


    def preprocess_output(self, outputs, coords):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        mouse_coords = (0, 0)
        gaze_coords = outputs[self.output_name[0][0]]

        angle_r_fc = coords[2]
        sin_r = math.sin(angle_r_fc * math.pi / 180.0)
        cos_r = math.cos(angle_r_fc * math.pi / 180.0)

        x = gaze_coords[0] * cos_r + gaze_coords[1] * sin_r
        y = -gaze_coords[0] * sin_r + gaze_coords[1] * cos_r

        mouse_coords = (x, y)

        return mouse_coords, gaze_coords
        
