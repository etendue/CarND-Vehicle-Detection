


## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/Histogram_bin_example.png
[image4]: ./output_images/Area_and_scale.png
[image5]: ./output_images/Find_Car_example.png
[image6]: ./output_images/Add_Heat_Threshold.png
[image7]: ./output_images/labels_map.png
[image8]: ./output_images/multi-vehicle-bbox.jpg
[video1]: ./project_video_output.mp4
[video2]: ./test_video_output.mp4 "test image from lession"

---
## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the **3rd code cell** of the IPython notebook `project5.ipynb` with function name `get_hog_features()`.  

I picked randomly some samples from  `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

`skimage.hog()` function accepts image with one channel. In order to explore different color channels, the `skimage.hog()` function needs to be called multiple times and features needs to be concatenated.

In later stage I tried different parameter combinations (`orient`, `pixels_per_cell`, and `cells_per_block`) with different `color space`. I grabbed randomly images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Grayscale` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

##### 1.1 binned color features, as well as histograms of color

I aslo applied the binned color features and histograms of color features for later SVM classifier. The code for this step is contained in **7th code cell**.
Here is example of those features for car image and non-car image.

![alt text][image3]


#### 2. Explain how you settled on your final choice of HOG parameters.

The first step to evaluate the choice of HOG parameters is the SVM classifier validation/accuracy score(which will be introduced in next chapter). This usually delivered a promising result.
It normally produced >97% accuracy. However the classifier is not very trustful when applied in images in `test_images` folder.

The criteria to settle the HOG parameters I applied is the performance on images in `test_images` folder and the `test_video.mp4`. Following parameters are not sensitive.
   
    * `orient`
    * `pixels_per_cell`
    * `cells_per_block`
im comparision to
    
    * color space (RGB,HLS,HLV,YCrCb,YUV) and color channel.

I tried RGB, HLS, YCrCB, YUV separately. And here are the experience with those:

    * RGB worked well in test_image in lesson video but turned out bad in this project 
    * S channel of HLS is supposed to be stable feature against lightness, but saturation is not a particular marker for distinguishing cars vs non cars
    * YCrCB(I haven't heard before) and just tried out
    * YUV with Y channel - recommend by some other students

The final result is preferred with  `YUV color space` with Y channel, although still not perfect. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVM`. First I defined a function to get the all features (color and HOG features) and converted to 1-D array in function `extract_features()` (**code cell 9**);
then I packed the job of training a classifier in function `train_svm_classifier()` in **code cell 10**. The function returns :

   * svc: classifier for car prediction
   * scaler: for feature normalization

Here is the main procedure:

   1. get images for cars and non-cars images files for training and validation 
   2. extract features
   3. do normalization use `StandardScaler`
   4. randomizing the train/validation sets
   5. train `LinearSVC` using train/validation data
   6. check the validation score
   
I got a **98.6%** accuracy for classification.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since the targets (cars) are scattered in the image captured from camera, to identify them by using the classifier, the image has to be splitted to small patches and resized to 64x64 size.
Function `slide_window` in **code cell 11**  is used to get all small rectangle patches by sliding the patch window from left to right, top to bottom on a image.
The `overlap` rate is used to calculate the stride/step size for sliding. The cars appear in different sizes in real 3D scene, therefore the `xy_window` is the patch window size. The ratio:
  
  * scale = xy_window[0]/64

is the `scale` in 1 dimension. 

Following image shows the applied the strategy for searching cars in area of interest and size of interest.


![alt text][image4]

In final step I fixed the `overlap` ratio to be **75%** in function `find_cars()` in **code cell 13**. This function (reference code from lesson) optimizes the HOG features extraction.
Instead to apply `hog()` on each patch window, the`hog()`is called once on the whole image and HOG feature vector is segmented according to patch size. The color features extraction is still done
 on every patch.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on **six** scales using YUV Y-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
Here are the parameters I adjust for the `find_cars` function:

| parameter        | defition   | value|
|:-------------:|:-------------:|:-----------:|
|colorspace     | Can be RGB, HSV, LUV, HLS, YUV, YCrCb|'YUV'|
|orient         | the orientation of HOG               |  9  |
|pix_per_cell   | pixel number per cell:HOG parameter  |  8  |
|cell_per_block | cell number per block: HOG parameter |  2  |
|hog_channel    | color channel for HOG;be 0, 1, 2, or "ALL"|0|
|spatial_bin_size |spatial bin size                    |  16 |
|color_bin_size | color bin size                       |  32 |
|scale          | patch window size / 64               | 1,1.25,1.5.175,2,2.5|

Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to test video result](./test_video_output.mp4)

Here's a [link to project video result](./project_video_output.mp4), the video contains the lane detection pipeline as well.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I introduced in the pipeline 4 parameters to minimize the false positive and smoothing the bounding box moving (**see code cell 18** )

| parameter        | defition   | value|
|:-------------:|:-------------:|:-----------:|
|frames_to_average |frame count to average the heat|5|
|frames_to_validate |frame count to validate, i.e. the video shall detect amount  of cars consectively to decide a new car is detected.|5|
|recognition_rate | the cars are only detected under certain scales. By observing the samples, 2 out of 6 scales patches are detected in average| 2|

With current SVM classifier, it performs bad when there is shadow of tree on the road, mainly the projection of leaves. But they are irregular and shapes changes quickly.
In function `validate_labels()`, I count the detected cars and compare to previous results, if there is a change then the labels are cached in a buffer. If the buffer reaches size
`frames_to_validate` the counts of cars are checked if they remain the constant value, it is considered to be new labels. In case of trees, the hot positive is filtered out due to the irregularity 
of detected count.

In function `multi_scale_find_cars_pipeline()` (**code cell 19** ). I recorded the positions of positive detections in each frame of the video by using multiple scales (6 scales in total) search method and 
use `add_heat()` to create a heatmap and  accumulate the result from different scales. Further more
 I also accumulate the "heatmapped" region from previous frames (parameter **frames_to_average** defines the size of this time window).

Finally I  thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


* Here are six frames and their corresponding `raw detected bounding box`,`heatmap`, `heatmap after threshold`, and `labelled image`:

![alt text][image6]



* Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major issue was to find the right parameter, e.g. the color space for HOG features and right algorithm to remove false positive.
It seems the classifier is quite sensitive with trained data. 

* In lesson the same train data is used, to be specific, part of train data is used. The test picture was well deteced with cars by using 'RGB' color space. See below picture.
* I used the same classifier to test on the test images of this project. The cars were hardly detected. 
* the classifier is good with  [`test_video.mp4` ](./test_video_output.mp4), but not perfect with the long video [`project_video.mp4` ](./project_video_output.mp4)

![alt text][image8]

I finally decided the color space 'YUL' which by luck performing relatively better, but still there are frames where no cars are detected, and for these with tree shadows,
plenty false positives are detected.

I think further improvement is to find proper parameters for robust classifier or find some robust classifier.
