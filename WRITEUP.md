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

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

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

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...