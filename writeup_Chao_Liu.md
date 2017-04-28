#**Project3 -- Behavioral Cloning** 

##Chao LIu

####This article is my solution to the Udacity Self-Driving Car Nanodegree Term 1 Project 3 -- Behavioral Cloning.

[//]: # (Image References)

[image1]: ./examples/NVIDIA_Net.png "Model Visualization"
[image2]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image3]: ./examples/recovery_1.jpg "Recovery Image"
[image4]: ./examples/recovery_2.jpg "Recovery Image"
[image5]: ./examples/recovery_3.jpg "Recovery Image"
[image6]: ./examples/original.jpeg "Normal Image"
[image7]: ./examples/flipped_image.jpeg "Flipped Image"
[image8]: ./examples/train_val_loss.png "Loss Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py -- containing the script to create and train the model
* drive.py -- for driving the car in autonomous mode
* model.h5 -- containing a trained convolution neural network 
* writeup_report.md -- summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
###Introduction
The object of this project is teaching a car to drive autonomously in a Udacity-developed driving simulator with deep learning algorithm. The simulator includes both training and autonomous modes. 

In training mode, user generated driving data is collected. The data includes the camera images which installed in the simulated car and the control informations (steering angle, throttle, brake, speed). With help of the Keras, a convolutional neural network (CNN) model is produced using the collected driving data and saved as `model.h5` .



---
###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The project instrucion from Udacity provides a known self-driving car model -- NVIDIA-Net. The diagram below is a depiction fo the NVIDIA network architecture.

![alt text][image1]
 
This network consists of 9 layers, including a normalization layer, 5 convolutional  layers and 3 fully connected layers . The input image is split into YUV planes and passed to the network.

The first layer of the network performs image normalization using a Keras lambda layer.

In the first three convolutional layers a strided convolution is used with 2x2 stride and  5x5 kernel, but in the last two convolutional layers, a non-strided convolution with 3x3 kernel is adopted. Following the 5 convolutional layers are the three fully connected layers leading to an output control value for steering.

The paper does not mention what kind of activation functions they used, so I choose the activation ReLU same as project 2.


####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 115). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and smooth around curves driving. 

Udacity provides a dataset that can be used alone to produce a working model. However, students are encouraged to collect our own. Here are some general guidelines for data collection:

- two or three laps of center lane driving
- one lap of recovery driving from the sides
- one lap focusing on driving smooth  around curves

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA network. I thought this model might be appropriate because it has 5 convolutional layers and 3 fully connected layers.

Secondly I should think about the image preprocess (model.py line 15 --26). The cameras in the simulator camera capture 160x320 images. Not all of these pixels contain useful information. In the image above, the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. In order to make the model to train faster, I crop each image to focus on only the portion of the image (top 50 pixels and bottom 20 pixels are cropped).  The input images feed to NVIDIA network are expected to be 66x200, so I need to resize each image of the collect data to 66x200 using the function `cv2.resize`. The NVIDIA network suggests the input image is split in YUV planes, but I found in the simulator enviroment the RGB feed images can do the better result. 

Next important mission is to implement a Python generator.  Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator we can pull pieces of the data and process them on the fly only when we need them. The function `generator` (model.py line 29 -- 59) is cloned from Udacity instruction. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added the dropout layers to the model.

Then I found some other strategies to combat overfitting and get the car to drive smoothly:

- Using the `L2` regularization to all model layers to replace the dropout layers
- Instead of the `ReLU` activation on fully connected layers, adding the `ELU` activation layer to all model layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I add the left and right camera images to feed the network.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py line 65 -- 101) is almost same as the NVIDIA network.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it is off on the side of road.  These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would a effective technique for helping with the left turn bias (model.py line 54 -- 55) For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

The NVIDIA network also use the multiple cameras, and luckly in Udacity simulator,  there are three cameras mounted on the car: a center, right and left camera. Using all three camera images to train the model is effective as recording recoveries. Feeding the left and right camera images to the model as if they were coming from the center camera can teach the model how to steer if the car drifts off to the left or the right. Here I use a parameter to correct the steering angle.

```python
	steering_center = float(row[3])
	correction = 0.2 # this is a correction parameter to tune
	steering_left = steering_center + correction
	steering_right = steering_center - correction
```


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the outputting of training and validation loss metrics. I used an adam optimizer so that manually training the learning rate wasn't necessary. Here is the relationship between the training and validation loss and epochs.

![alt text][image8]
