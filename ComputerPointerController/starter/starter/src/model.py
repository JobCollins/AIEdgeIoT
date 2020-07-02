'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.6):
        self.model_name = None
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
        self.network = elf.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self):
        pass

    def check_model(self):
        pass

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        w, h = self.input_shape[3], self.input_shape[2]
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        pass

    def wait(self):

        return self.network.requests[0].wait(-1)
