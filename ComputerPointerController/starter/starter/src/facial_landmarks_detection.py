from model import Model_X

class Landmark_Detection(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model_X.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Facial Landmarks Detection'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))

    
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pred_img = self.preprocess_input(image)
        self.network.start_async(
            request_id=0, inputs={self.input_name: pred_img}
        )

        left_eye, right_eye, eye_coords = [], [], []

        if self.wait() == 0:
            outputs = self.network.requests[0].outputs[self.output_name]
            left_eye, right_eye, eye_coords = self.preprocess_output(outputs, image)

        return left_eye, right_eye, eye_coords


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        w, h = image.shape[1], image.shape[2]
        outputs =  outputs[0]
        
        left_eye_xmin = int(outputs[0][0][0] * w) - 10
        left_eye_ymin = int(outputs[1][0][0] * h) - 10
        right_eye_xmin = int(outputs[2][0][0] * w) - 10
        right_eye_ymin = int(outputs[3][0][0] * h) - 10

        left_eye_xmax = int(outputs[0][0][0] * w) + 10
        left_eye_ymax = int(outputs[1][0][0] * h) + 10
        right_eye_xmax = int(outputs[2][0][0] * w) + 10
        right_eye_ymax = int(outputs[3][0][0] * h) + 10

        left_eye , right_eye, eye_coords = [], [], []

        left_eye = image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
        right_eye = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
        eye_coords = [[left_eye_xmin, left_eye_ymin, left_eye_xmax, left_eye_ymax], [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]

        return left_eye, right_eye, eye_coords