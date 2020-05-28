# Project Write-Up


## Explaining Custom Layers

The process behind converting custom layers involves...
1. Feeding the target model to the Model Optimizer, which loads the model.
2. The Model Optimizer then extracts information for each layer in the above model. Here, the Custom Layer Extractor extension identifies the custom layer operation and extracts the parameters for each instance of the custom layer.
3. The model is then optimized.
4. The optimized model is output as model IR files, thede files are used in the inference engine to run the model.

Some of the potential reasons for handling custom layers are...
1. The custom layers are not in the list of supported layers.
2. To allow you to plug in your own implementation for existing or completely new layers.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

For inference time I used the time() module and calculated the difference between time before inference and after inference. A python notebook *ssd mobilenet_v2 inferencing* is included.
1. The model .pb file took just over *0.05 seconds* to run inference.
2. The IR model file took just over *0.07 seconds* to run inference.

For the size comparison, the .pb file is *66.4 MB* while the IR .bin file is *64.2 MB*. That is over 2 MB lighter.

For the model accuracy;

1. I computed the precision and recall values for the top-3 predictions. Out of the top-3 predictions, I picked those that were correct *x*.
2. Worked out the precision . 
3. Then the recall.
4. Repeated this process for the images' the top-3 predictions. This process yields a list of precision values.
5. The next step  I computed the average for all the top-3 values to get the Average Precision (AP). 
6. Computed the mean of the APs for each class, giving us a mAP for each individual class, and averaged them together to yield mAP of the model.

The IR model had a mAP value of 23 while the pre-IR model had a value of 25.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Evaluating service delivery and efficiency in service industry. These can include cashiers in stores, teller in banks, counters in government services hubs e.g.Huduma Centres in Kenya, or tickets sale at movie theaters. Insight from the app on duration can tell how much time a customer spends at a counter during service, can help operations improve efficiency by helping them decide whether more counters/tellers are needed to ease waiting times. The same insight can help identify which services offered take more times as compared to other services in a service hub.

The same application can be used to evaluate how effective an ad or marketing campaign is. Especially, in a marketing expo. The count of number of people visiting a stand, and the amount of time they spent there can inform whether a marketing strategy is effective or not, is it catchy?

Another place the app can really be helpful is surveillance in restricted areas. Unusually long times (duration) spent by the same person (detection) in such an area can help flag potential security risks.

In retail stores, the application of this app can help identify which shelves are mostly visited by customers. This would mean more stocking of items in that shelves, perhaps, in various times of the month or year.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

While ssd mobilenet is able to detect, and classify objects in low lighting, an object that is moving closer to the camera - in low lighting - will be detected/classified at a lower model accuracy or even not detected at all. This means in cases where low lighting is present an object might not detected depending on how close it is to the camera. Therefore, safe to say, an accurate classification can be found for the detection of objects and facial recognition at varying distances.

