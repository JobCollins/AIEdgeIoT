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
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None
        self.network = None
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core = IECore()


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, left_eye, right_eye, coords, request_id=0):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)
        self.network.start_async(request_id, inputs={
            'left_eye': left_eye,
            'right_eye': right_eye,
            'head_pose_coords': coords
            })

        if self.wait() == 0:
            outputs = self.network.requests[0].outputs
            mouse_coords, gaze_coords = self.preprocess_output(outputs, coords)

        return mouse_coords, gaze_coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        w, h = self.input_shape[3], self.input_shape[2]
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, h, w)

        return image

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
        
