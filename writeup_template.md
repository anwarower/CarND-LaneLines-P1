# **Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./writeup_images/coverPhoto.png "cover"
[image3]: ./writeup_images/laneChange.png "Lane Change"
[image4]: ./writeup_images/dashed.png "Dashed next to Solid"
[image5]: ./writeup_images/curve.png "Curve"
![alt text][image2]


### I. Pipeline Description

The processing of a single frame consisted of defined pipeline stages. Those can be described as the following.
#### 1. Grayscale Transformation
This step is necessary for the coming steps.
#### 2. Smoothing Filter
For this I used a Gaussian Filter with a window length of 7. The purpose of this step is to suppress the noisy edges in the image.
#### 3. Raw Edge detection using Canny Edge Detector
Edges are the key features in the image. They are defined by a high drop or rise in intensity. To extract the edges out of the image, Canny Edge Detector has been used with a low threshold of 50 and a high threshold of 150.
#### 4. Region of Interest Masking
Based on inspection and trail and error, a polygon has been defined in the lower half of the image as a region of interest.
#### 5. Edge linking usig Hough Transform
Among the detected edges in the last step, we filter those ones that contribute to prominent lines. This is achieved using the Hough Transform.
#### 6. Merging of Hough Lines and Inter-Frame Smoothing
In order to draw a single line on the left and right lanes, the draw_lines() function had to be modified.
Instead of just outputting each line detected by the Hough Transform, a single unified line has been calculated. The relevant method for this functionality is `merge_lines(side, lines)`.

The calculation of the unified lane line has been achieved by two different ways which could be described as follows:
###### * Line Averaging
This method has been used when no history is available yet from previous frames. In this case, the lines have been split using their slopes to right and left candidates respectively. Afterwards, for each side, the parameters of all the candidates have been averaged together to form the unified edge. This method is implemented in `get_averaged_line(side, lines)`.  

###### * Line Prediction and Projection  
[comment]: <> (![alt text][image1])
In case enough frames have been already processed and a stable history is available, a heuristic has been used that expects the new output to be pretty close to the previous output. Using the function `get_best_twin(side, lines)`, the closest Hough line found is chosen, that fits best the history.


After the unified line is calculated in a current frame, its parameters are inserted into a buffer of a length N containing all outputs of the last N frames. The output of the current frame is then an average of the entries in this buffer. A buffer entry for a given frame is a slope and intercept, which correspond to the line coefficients.

#### 7 Line Exterpolation
For the line equation we derived in the previous steps we would like to define the convenient *range*. This means basically the two values *yUp* and *yDown* defining where a lane marker starts from the bottom and how high it gets. I always assumed that the lane markers start from the image base. This means *yDown* was set to the height of the input image (origin is on the upper left corner). To find yUp, the distance between the two lane markers is calculated at two arbitrary y values. Afterwards, a linear relation between this distance and y is estimated. We select then the yUp value corresponding to a (distance = 50 pixels) between the left and right lane markers. In case the marker is found on only one side, yUp is just the upper boudary of the region of interest (See `intersect_at_given_distance(lineR, lineL, distance)` and `extrapolate_line(img, line)`).     

### II. Potential shortcomings with the current pipeline

#### 1 Lane Change Scenarios
One potential shortcoming could happen when there is a lane change scenario. This is because we do not have a motion compensation mechanism that fixes the inconsistency between our prediction and the current frame.  
![alt text][image3]

#### 2 Solid markers that exist next to the inner dashed markers
Sometimes the algorithm is tempted favour solid lanes over dashed lanes, even if the solid lanes are further. This is because solid lanes are dominant in the Hough Transform.
![alt text][image4]

#### 3 Curve situations
Since we are modelling the lane markers as lines, we have no capability of perfectly locatng curved lines. Nevertheless, in many of the cases, the algorithm tracks the tangent to this curve almost correctly.
![alt text][image5]

### III. Suggestion for possible improvements to the pipeline

#### 1 A Map to 3D World
We need a mathematical model, that calibrates our camera, localizes it and maps the 2D measurements we are working with to 3D coordinates. This would enable us to perform more complex plausibility measures and achieve better tracking using motion compensation. This is actually not an option if this algorithm is to be used for autonomous driving.

#### 2 An advanced model for lane markers
Sometimes lines are too simple to model lane markers. A higher order polynomial would be more discriptive, nevertheless it would be much more prone to errors.





# **IMPORTANT!**
# If you are interested to see extra examples for the performance of this algorithm, you can check it [Here](https://www.youtube.com/playlist?list=PLzCBDDtp2hMdCsoM-LQYECuMvt1MW6CJ8)
