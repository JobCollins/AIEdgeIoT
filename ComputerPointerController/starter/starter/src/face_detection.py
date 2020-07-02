'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import sys
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore

from model import Model_X

class Face_detection(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.5):
        '''
        TODO: Use this to set your instance variables.
        '''
        Model_X.__init__(self, model_name, device='CPU',extensions=None, threshold=0.6)
        self.model_name = 'Face Detection'
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
            outputs = self.network.requests[0].outputs[self.output_name]
            bbox, image_copy = self.preprocess_output(outputs, image)

        return bbox, image_copy

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