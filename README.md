# CAR OBJECT DETECTION
### We have created an object detection model that can accurately detect vehicles in a video frame or an image using the predefined architectures in the TensorFlow Object Detection API
![tf](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png)



ensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.



## Summary

- Install Tensorflow
- ---
- Install TFOD dependencies (Protobuf, Cocobuf API )
- ---
- convert csv labels to tf.record file format
- Complile and Install Object detection API in /models/research path 
- Download and extract pretrained model
- Train the model
- Activate TensorBoard

> model setup and training were
> all run in Google Colab
> ![N|Solid](data:image/png;base64,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)


## TendorBoard Metrics
DirectionBoxes recall
>- ![DirectionBoxes recall](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/DirectionBoxes%20recall.PNG?raw=true)
Direction Boxes Precision
- ![DirectionBoxes Precision](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/DirectionBoxes_Precision.PNG?raw=true)
Evaluation
- ![eval 0](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/eval%200.PNG?raw=true)
- ![eval 1](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/eval%201.0.PNG?raw=true)
- ![eval 2](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/eval%202.0.PNG?raw=true)
Learning step and steps per second
- ![learning step and steps per second](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/learning%20step%20%20and%20steps%20per%20secs.PNG?raw=true)
Loss One
- ![loss functions](https://github.com/jean-on-hub/Vehicle-detection-model/blob/main/tensorboard%20snippets/loss%20functions.PNG?raw=true)
Loss Two
- ![loss](https://raw.githubusercontent.com/jean-on-hub/Vehicle-detection-model/80d3954bc517415963cd1da91da8aa8943727998/tensorboard%20snippets/loss2.PNG)




