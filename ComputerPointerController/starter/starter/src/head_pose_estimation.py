from model import Model_X

class Head_pose(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_path, device='CPU', extensions=None, threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model_X.__init__(self, model_path, device, extensions, threshold)
        self.model_name = 'Head Pose Estimation'
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pred_img = self.preprocess_input(image)
        self.network.start_async(
            request_id=0, inputs={self.input_name: pred_img}
        )

        if self.wait() == 0:
            outputs = self.network.requests[0].output_shape
            output = self.preprocess_output(outputs)

        return output


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        processed_output = []

        processed_output.append(outputs['angle_y_fc'][0][0])
        processed_output.append(outputs['angle_p_fc'][0][0])
        processed_output.append(outputs['angle_r_fc'][0][0])

        return processed_output


