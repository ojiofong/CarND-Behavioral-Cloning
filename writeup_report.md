#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_visualization.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/flipped0.jpg "Normal Image"
[image7]: ./examples/flipped1.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven 
autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution 
neural network. The file shows the pipeline I used for training and validating 
the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 1 Lambda layer, 1 Cropping2D Layer, 5 Convolution2D layers, 1 Flatten layer and 4 Dense layers. 
Each convolution is followed by a SpatialDropout2D. 

I compiled the model using Adam optimizer with a learning rate 0.0001. 
This very small learning rate allows me to get a generalized result on 
a small value of 6 Epochs.

The model includes ELU activations to introduce non-linearity, 
The data is conveniently normalized and processed by the network via Keras' 
Lambda layer and Cropping2D layer. The Lambda layer normalizes each training image, 
while the Cropping2D layer crops off unneeded pixels from each image. 

####2. Attempts to reduce overfitting in the model

Overfitting occurs when a model memorizes the training data and is not 
able to generalize well on new test data. To mitigate this I split off 
20% for validation, applied shuffling and introduced SpatialDropout2D 
regularization after each convolution.

The model was trained and validated on different data sets to ensure 
that the model was not overfitting. The model was tested by running it 
through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used 
a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia paper: 
 http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf. 
 I thought this model might be appropriate because it is popularly known to work well and 
 also it is not too complex.

In order to gauge how well the model was working, I split my image and 
steering angle data into a training and validation set. 
I found that my initial model had a low mean squared error on the training set 
but a high mean squared error on the validation set. 
This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that every Convolution2D Layer 
is followed by a SpatialDropout2D regularization.

In addition to the Dropout, I split off 20% of the training data for validation, 
applied shuffling while training the data using Keras model.fit API.

The final step was to run the simulator to see how well the car was driving around 
track one. There were a few spots where the vehicle fell off the track. Usually at 
sharp bends or turns. Additionally the vehicle is more likely turn off track in 
a one direction.

To improve the driving behavior in these cases, I did not include any data with a 
steering angle close to zero. In other wards I ignored data with very low steering angles. 
That way the models is more adept to learning to make turns. I also augmented the data 
by adding a flipped image and corresponding flipped steering angle for every image in the 
training sample so as to balance learning to turn left and right. Furthermore, I added 
a correction factor of +0.1 and -0.1 to the steering angles of the left and right camera 
images respectively. This helps the model learn to steer a bit back to the middle of the road.

At the end of the process, the vehicle is able to drive autonomously around the track 
without leaving the road.

####2. Final Model Architecture

My model consists of a convolution neural network with 1 Lambda layer, 1 Cropping2D 
Layer, 5 Convolution2D layers, 1 Flatten layer and 4 Dense layers. Each convolution 
s followed by a SpatialDropout2D.

I compiled the model using Adam optimizer with a learning rate 0.0001. This very 
small learning rate allows me to get a generalized result on just 6 Epochs.

The model includes ELU activations to introduce non-linearity, The data is 
normalized and processed in the model using a Keras Lambda layer and 
Cropping2D layer which conveniently delegates tasks to the network. 
The Lambda layer normalizes each training image, while the Cropping2D 
layer crops off unneeded pixels from each image. Hence reducing noise while increasing 
the training speed.

Here is a visualization of the architecture:

![alt text][image1]

I used the Cropping2D Layer to crop/reshape my original input shape into (66, 200, 3) 
to match the Nvidia model's input shape. Here's a summary of the model outputted by 
the `model.summary()` function

````
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 66, 200, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 100, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
spatialdropout2d_1 (SpatialDropo (None, 33, 100, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 50, 36)    21636       spatialdropout2d_1[0][0]         
____________________________________________________________________________________________________
spatialdropout2d_2 (SpatialDropo (None, 17, 50, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 23, 48)     43248       spatialdropout2d_2[0][0]         
____________________________________________________________________________________________________
spatialdropout2d_3 (SpatialDropo (None, 7, 23, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 21, 64)     27712       spatialdropout2d_3[0][0]         
____________________________________________________________________________________________________
spatialdropout2d_4 (SpatialDropo (None, 5, 21, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 19, 64)     36928       spatialdropout2d_4[0][0]         
____________________________________________________________________________________________________
spatialdropout2d_5 (SpatialDropo (None, 3, 19, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3648)          0           spatialdropout2d_5[0][0]         
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           364900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 501,819
Trainable params: 501,819
Non-trainable params: 0
____________________________________________________________________________________________________

````

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using 
center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the 
road back to center so that the vehicle would learn to stay on the road instead 
and not drive off the road
These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5] 

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this 
would help the model to balance learning to turn left and right.
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

Finally, I decided to use the training data provided by Udacity.
After the collection process, which did not include data with low steering 
angles, I had 11,025 number of data points. This number doubled 
to 22,050 after augmentation because flipped images were added. 
I then preprocessed this data by converting the images from BGR to RGB color channels. 
I delegated normalizing and cropping the image to the network by utilizing Keras' Lambda 
and Cropping2D Layers. Afterwards, I conveniently used Keras' model.fit API to handle shuffling and 
validation set splitting. I used 20% of the data for validationi.

I then used 80% of the sample to train the model. The validation set helped 
determine if the model was over or under fitting. The ideal number of epochs 
was 6 as evidenced by the fact that the training and validation loss was not 
significantly improving. I used an adam optimizer so that manually training the 
learning rate wasn't necessary.
