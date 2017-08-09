# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 the recording of my autonomous run

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 71-82).  It was implemented by following the guidelines laid out by NVIDIA for their behavioral cloning neural net.
The model includes RELU layers to introduce nonlinearity (code line 71 - 75), and the data is normalized in the model using a Keras lambda layer (code line 68) and cropped using a Keras Cropping2D layer (code line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 77 and 81). 

The model was trained and validated on different data sets to ensure that the model was not overfitting by using two different image generators based on different sample batches (code line 54 and 55). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving reverse around the track.  I made extra correction runs in the areas where the shoulder went from a curbed barrier to a sand shoulder as my initial attempts at navigating the track would regularly fail in such areas.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the NVIDIA proven neural net to solve the specific problem at hand. Some tweaking was done in the normalization and cropping of the data to reduce the noise produced by the background and hood that could be seen in every image. I also introduced dropout layers early in the design phase to prevent overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. From the beginning, my model's mse on both the training and validation sets reduced together showing that I was able to prevent overfitting.

The final step was to run the simulator to see how well the car was driving around track one. After the first run it was obvious I needed more data as my car quickly veered off of the track. With this knowledge I ran through the simulator a few more times focusing on another center lane pass, a reverse center lane pass, and then corrective passes over problem areas which I performed by starting the recording as the vehicle was situated near the edge of the track and recorded the correction back to teh middle of the track. I then ran data through the neural net to further train my model and re-ran the simulator to see how the car performed. At that point, the vehicle was able to make numerous circuits around the track without leaving the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final architecture of the neural net can be seen below.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I ended up recording two laps on track one using center lane driving. I then recorded a reverse lap on track one using center lane driving. Finally, I recorded the vehicle recovering from the left and right sides of the road back to the center so the vehicle would learn how to correct itself in such situations and that given the situation that we are too close to an edge, it should move back to center.

To augment the data sat, I also flipped images and angles thinking that this would give me a more generalized data set to train on.  In the end I found that it did little to generalize improve the neural nets performance and the augmentation was stripped from the final model and an image generator was added in its place to perform fit_generator fitting through Keras to manage memory limitations on my machine.

I then split the data into two subsets, a training set and a validation set using sklearn's train_test_split method breaking off 20% of the data as a validation set. 

I then created two data generators, one for testing and one for validation, that would shuffle the samples and the batches for their passes.

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a low change in mse between epochs at that point. I used an adam optimizer so that manually training the learning rate wasn't necessary.
