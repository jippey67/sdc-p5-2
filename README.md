# Vehicle Detection and Tracking

This repository contains the work I did for the vehicle detection and tracking project in the Udacity Self-Driving Car Nanodegree program. The objective of the project is to find cars in the images of a front looking camera, and track these cars. Goal is to find all cars as soon as they enter into view, while minimizing the chance of the occurence of 'false positives': Find a car where there isn't one. 

First a car finding algorithm has to be set up. This involves obtaining a collection of images of cars and non cars, and training a classifier that succesfully finds cars in images. With the classifier cars can be found in the image provided by the forward looking camera. As cars can be in different spots in an image and of different sizes, a sliding window technique with varying window sizes needs to be implemented. To correct for finding (too many) false positives and to build a smooth view on the surrounding cars, an averaging and thresholding method is used. In the project I made use of many Python functions provided in the Udacity SelfDriving Car NanoDegree lessons.

## Training data

Within the project a couple of data sources are available. I chose to combine all of them to have a broad collection of cars and notcars. I ended up with a set of 8792 images of cars and a set of 8968 images that contain images of roads and surroundings, without cars in them. Both sets are in RGB color space and of format 64x64 pixels. They are also of  approximatelythe same size, so no augmentation of one of the calsses is required.
Below are 5 examples of cars en 5 examples of non cars. 

<img src="https://github.com/jippey67/sdc-p5-2/blob/master/images/1.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543698/0a8d830e-e932-11e6-9c46-fb242c1301d0.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543699/0a94c9de-e932-11e6-819e-be601985963d.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543700/0a959c74-e932-11e6-950f-0139c10eb307.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543789/5dc50182-e932-11e6-98e7-15e9a6b55b81.jpg" width="128" height="128" /> 

<img src="https://cloud.githubusercontent.com/assets/23193240/22543711/133bc4fc-e932-11e6-93e6-50b527f17432.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543713/133f3f2e-e932-11e6-8b7e-887529ea55e3.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543712/133db17c-e932-11e6-8737-c0a4c630509f.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543714/133f54aa-e932-11e6-8dd0-370e2d227a79.jpg" width="128" height="128" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22543796/6819e67a-e932-11e6-8863-f57416bd8b7e.jpg" width="128" height="128" /> 

The data were labeled "1" for cars and "0" for non cars, and after creating a random sequence, split into 80% training data and 20% test data. The training data were fed to various collections of feature extractors - as described below - and afterward the feature values were normalized inorder to prevent one feature dominating the others.

## Training a classifier

For this project I chose to use a Support Vector Machine with a linear kernel. As there were many others parameters to tune I sticked with the standard parameters for the SVM, as they worked quite good from the beginning.
A couple of features are available to implement with this classifier: spatial binning of color, color histograms and histograms of oriented gradients (HOG). Each of these involves the selection of parameters. As especially the HOG feature has many parameters, I decided to investigate this one first with a simulation in which I varied a couple of parameters. In a following step I combined the three features and ran another simulation to arrive at parameters to use in the video pipeline.

### The HOG parameters

Preliminary research already proved the RGB wasn't very useful for using the HOG feature. HSV did a much better job, so I conducted the HOG research within this colorspace. Parameters varied for HOG were:
* layer within the colorspace (could also be all three together)
* the number of orientation bins
* number of pixels per cell
* number of cells per block

It became immediately clear that training an SVM on only HOG with only one layer of the colorspace provided much worse results than when using all layers. For brevity I left out the results on single layers. The results for the three layers combined is show in the table below: 

**# orientations**|**pix/cell**|**cells/block**|**HOG channel**|**test accuracy**
:-----:|:-----:|:-----:|:-----:|:-----:
13|16|2|ALL|0.9937
13|16|1|ALL|0.9900
**13**|**8**|**2**|**ALL**|**0.9986**
**13**|**8**|**1**|**ALL**|**0.9977**
13|4|2|ALL|0.9968
**13**|**4**|**1**|**ALL**|**0.9991**
9|16|2|ALL|0.9950
9|16|1|ALL|0.9887
9|8|2|ALL|0.9937
9|8|1|ALL|0.9955
**9**|**4**|**2**|**ALL**|**0.9977**
9|4|1|ALL|0.9959
5|16|2|ALL|0.9900
5|16|1|ALL|0.9855
5|8|2|ALL|0.9946
5|8|1|ALL|0.9932
5|4|2|ALL|0.9950
5|4|1|ALL|0.9941

In bold the highest accuracies. Trying to run the training on an even larger number of orientations wasn't succesful because of an unacceptable long training time. 

The results are very promising with such high accuracy. However the car images used are always completely filling the image. In scanning over the images in the video pipeline this will certainly not always be the case. So to make the classifier more general I wanted to include spatial binning and color histograms as well. Including all three HOG image layers in a combined search wasn't workable as that consumed too much processing power. I decided to just use one HOG layer instead (layer 0), in the further research. 

### training the combinations of features

In this next step (three feature optimalization) I also considered other color spaces, to obtain a feeling for what they do when used with the various features. The table below shows results for imagesize 64x64, 64 bins for histograms:

**color space**|**spatial color binning**|**histograms of color**|**HOG**|**average test accuracy**|**standard deviation**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
HSV|FALSE|FALSE|TRUE|0.9800|0.0025
**HSV**|**FALSE**|**TRUE**|**FALSE**|**0.9972**|**0.0013**
HSV|FALSE|TRUE|TRUE|0.9961|0.0019
HSV|TRUE|FALSE|FALSE|0.9188|0.0052
HSV|TRUE|FALSE|TRUE|0.9908|0.0020
HSV|TRUE|TRUE|FALSE|0.9855|0.0020
**HSV**|**TRUE**|**TRUE**|**TRUE**|**0.9974**|**0.0012**
LUV|FALSE|FALSE|TRUE|0.9777|0.0038
LUV|FALSE|TRUE|FALSE|0.9911|0.0008
LUV|FALSE|TRUE|TRUE|0.9952|0.0013
LUV|TRUE|FALSE|FALSE|0.9410|0.0051
LUV|TRUE|FALSE|TRUE|0.9893|0.0024
LUV|TRUE|TRUE|FALSE|0.9718|0.0031
LUV|TRUE|TRUE|TRUE|0.9935|0.0017
HLS|FALSE|FALSE|TRUE|0.9775|0.0037
HLS|FALSE|TRUE|FALSE|0.9950|0.0011
HLS|FALSE|TRUE|TRUE|0.9966|0.0012
HLS|TRUE|FALSE|FALSE|0.9218|0.0059
HLS|TRUE|FALSE|TRUE|0.9902|0.0015
HLS|TRUE|TRUE|FALSE|0.9802|0.0028
HLS|TRUE|TRUE|TRUE|0.9954|0.0016
YUV|FALSE|FALSE|TRUE|0.9805|0.0030
YUV|FALSE|TRUE|FALSE|0.5454|0.0123
YUV|FALSE|TRUE|TRUE|0.9795|0.0024
YUV|TRUE|FALSE|FALSE|0.9376|0.0046
YUV|TRUE|FALSE|TRUE|0.9893|0.0020
YUV|TRUE|TRUE|FALSE|0.9394|0.0059
YUV|TRUE|TRUE|TRUE|0.9901|0.0017
YCrCb|FALSE|FALSE|TRUE|0.9778|0.0023
YCrCb|FALSE|TRUE|FALSE|0.5433|0.0083
YCrCb|FALSE|TRUE|TRUE|0.9792|0.0026
YCrCb|TRUE|FALSE|FALSE|0.9366|0.0045
YCrCb|TRUE|FALSE|TRUE|0.9901|0.0022
YCrCb|TRUE|TRUE|FALSE|0.9395|0.0035
YCrCb|TRUE|TRUE|TRUE|0.9904|0.0021

The highest accuracy is scored in the HSV color space, using all three features. Interesting is that using only the spatial color binning feature also provides a high accuracy. Both rows are shown in bold in the table above. Looking over the various color spaces it shows that many of them score quite good when all three features are combined. 

What settings to choose? The highest accuracy is within the HSV space with all features combined. This was the first combination used in finding cars in the test images and the video pipeline. As the proof of the pudding is in the eating (and not in the test accuracy), I also tried the other color spaces to find cars in the test_images and the video. Using the YCrCb color space provided the best results there, and that is what I finally settled on.


## Sliding window search

Cars appear smaller in the image the farther they are away. The range of the image to be searched for small instances of a car is relatively small as is the size of the car image. From a couple of pictures it appears that small cars fit in a box of 32x32 pixels, whereas nearby and therefore larger cars need up to 128x128 pixels. In general there are no cars to be expected in the range y < 400. 

Another aspect is the overlap of the sliding windows. Although with an overlap of 50% most car objects were found in the images, after some experimentation I settled on an overalp of 75%. This resulted in the cars often found multiple times in an image, whereas the false postives just occurerd once for most of the time. This made it possible to filter out many false positives by applying a threshold. I finally setteld on these sliding window settings:

**type of car**|**size of box**|**y-range**|**x-range**|**step size**|**frames in y direction**|**frames in x direction**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
small|64x64|400 - 496|320 - 960|75%|3|37
medium|96x96|400 - 544|320 - 960|75%|3|20
large|128x128|400 - 592|192-1088|75%|3|25
 | | | | | | 
total number of frames| | | | |246| 

The two pictures below show found cars and false positives

<img src="https://cloud.githubusercontent.com/assets/23193240/22651609/f36adc08-ec83-11e6-9c6a-b2d773c8811f.png" width="400" height="200" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22652640/7fcfd038-ec87-11e6-9b0d-e3fe639b8a43.png" width="400" height="200" /> 

To filter out as many false positives as possible I used a two step filtering process. In the first step the 'heat' of the analysed image was determined. As the cars are detected in a multiple of the sliding windows, a lot of heat is generated wher the cars are. It is anticipated that false positives generate less heat. So when the heat was smaller or equal to 2, this was marked as a false positive and left out of the heat of the image at hand.
The next step was an averaging mechanism with another filter. All heat maps were fed to a queue of length 10. All layers in the queue are then added up. Pixels with some heat, but below or equal to threshold of 4 are once again considered to be potential false positives, and are therefore removed.

Below are two images of the road and the heat generated (in green).

<img src="https://cloud.githubusercontent.com/assets/23193240/22664345/869338f0-ecb0-11e6-938c-46dd89b11989.png" width="400" height="200" /> 
<img src="https://cloud.githubusercontent.com/assets/23193240/22664344/86928536-ecb0-11e6-8f29-822d956f1147.png" width="400" height="200" /> 

## Python programs used in this project

I split the training of the model and the actual pipeline into two different Python programs:

car_finder.py was used to train the model. It includes various steps for analyzing what various parameters do to the accuracy of the model. As a final step, it saves the chosen model alongside with the feature scaler.

pipeline.py is where the actual work on the video was done. It loads the model and scaler. The windows in every frame are fed into the scaler and afterwards into the classifier. The consecutive steps are the measurement of heat and the building of a heat queue of the last 20 frames. That is where the filtering of false positives is done (row 119 - row 127). Based on that information, the boxes where the cars are assumed to be, are drawn on the original image.

A third Python file 'lesson_functions.py' is included. This contains a couple of files used in the course, and in the above two programs.

## The result

The whole pipeline led to the creation of this [video](https://github.com/jippey67/sdc-p5-2/blob/master/boxed_videop.mp4). It tracks the cars in the neighborhood very well. And there is the occasional false positive on the left side. I could have easily left that one out by starting the sliding windows more towards the center of the image, but as that would deteriorate the generality of the pipeline I decided to keep it in.

## Discussion

* The pipeline generally works well. But the incidental occurence of false positives requires some improvement before the pipeline can be put to actual use. Using a deep learning approach with a convolutional network to detect cars might improve the results over those of the SVM used in this project.

* The processing time required by the pipeline was around half a second per frame. This prohibits the use in real time as 24 frames per second need to be processed then. Programming in C++ would certainly speed up the process, but other improvements might be needed as well.

* The selection of workable parameters took a lot of time. Many 'heavy' combinations led to a hanging processor. But playing with the various possibilities certainly increased my knowledge how they influenced the final results. And as in earlier projects: A metric like accuracy is a good guidance, but what counts is the final result. 

## sources
* udacity.com: various Python functions
