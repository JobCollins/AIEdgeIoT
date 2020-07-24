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

        l_eye, r_eye, eyes = [], [], []

        if self.wait() == 0:
            outputs = self.network.requests[0].outputs[self.output_name]
            l_eye, r_eye, eyes = self.preprocess_output(outputs, image)

        return l_eye, r_eye, eyes


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        w, h = image.shape[1], image.shape[0]  #converted to int
        outputs =  outputs[0]
        
        l_eye_xmin = int(outputs[0][0][0] * w) - 10
        l_eye_ymin = int(outputs[1][0][0] * h) - 10
        r_eye_xmin = int(outputs[2][0][0] * w) - 10
        r_eye_ymin = int(outputs[3][0][0] * h) - 10

        l_eye_xmax = int(outputs[0][0][0] * w) + 10
        l_eye_ymax = int(outputs[1][0][0] * h) + 10
        r_eye_xmax = int(outputs[2][0][0] * w) + 10
        r_eye_ymax = int(outputs[3][0][0] * h) + 10

        l_eye , r_eye, eyes = [], [], []

        l_eye = image[l_eye_ymin:l_eye_ymax, l_eye_xmin:l_eye_xmax]
        r_eye = image[r_eye_ymin:r_eye_ymax, r_eye_xmin:r_eye_xmax]
        eyes = [[l_eye_xmin, l_eye_ymin, l_eye_xmax, l_eye_ymax], [r_eye_xmin, r_eye_ymin, r_eye_xmax, r_eye_ymax]]

        return l_eye, r_eye, eyes