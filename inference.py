#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    
    cpu_extension = "/opt/intel/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.core = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.ex_net = None
        self.async_req = None
        

    def load_model(self, model, device_name="CPU", extension=cpu_extension):
        ### TODO: Load the model ###
        self.core = IECore()
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.network, device_name=device_name)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers)!=0:
            print("These are the unsupported layers: {}".format(unsupported_layers))
        ### TODO: Add any necessary extensions ###
        self.core.add_extension(extension, device_name)
        ### TODO: Return the loaded inference plugin ###
        self.ex_net = self.core.load_network(self.network, "CPU")
        print("IR loaded successfully")
        ### Note: You may need to update the function parameters. ###
        
        return self.ex_net

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob  next(iter(self.network.outputs))
        
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        self.async_req = self.ex_net.start_async(
            request_id = 0, inputs = {self.input_blob: image}
        )
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.ex_net.requests[0].wait(-1)

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        output = self.ex_net.requests[0].outputs[self.output_blob]
        return output
