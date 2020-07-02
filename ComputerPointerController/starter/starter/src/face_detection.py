'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Face_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
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
            outputs = self.network.requests[0].outputs[self.output_name]



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

    def preprocess_output(self, coords, image):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        w, h = int(image.shape[1]), int(image.shape[0])
        bbox = []
        image_copy = image
        coords = np.squeeze(coords)

        for coord in coords:
            image_id, label, threshold, xmin, ymin, xmax, ymax = coord
            
            if label==1 and threshold >= self.threshold:
                xmin = int(xmin * w)
                ymin = int(ymin * h)
                xmax = int(xmax * w)
                ymax = int(ymax * h)
                bbox.append([xmin, ymin, xmax, ymax])
                image_copy = image[ymin:ymax, xmin:xmax]

        return bbox, image_copy

    def wait(self):
        
        return self.network.requests[0].wait(-1)