
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import math
from IPython.core.debugger import set_trace
import os


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
    
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
 
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ) 


# In[4]:


#global variables 
BUFFER_SIZE = 25; 
globalR = np.zeros((BUFFER_SIZE, 2))
globalL = np.zeros((BUFFER_SIZE, 2))
frameNum = 0
beliefAge = 0


# In[5]:


#judge if two lines are close enough or if they define a jump 
def check_Fit(line1, line2, slopeThreshold = 0.1, interceptThreshold = 25):
    slopeOK     = (abs(line1[0] - line2[0]) < slopeThreshold);
    interceptOK = (abs(line1[1] - line2[1]) < interceptThreshold)
    return (slopeOK & interceptOK);

#update the history used for inter-frame smoothing 
def update_Buffer(newV, side): #0, 1 Left, Right
    global frameNum 
    global globalR
    global globalL 
    global BUFFER_SIZE
    global beliefAge
    index = frameNum % BUFFER_SIZE
    
    #prevent bad guys from ruining the buffer
    if frameNum > BUFFER_SIZE:
        smooth = get_Smoothened(side);
        if(smooth['success']):
            currAvG = smooth['smoothened']
            if(check_Fit(currAvG, newV)==False):
                if(beliefAge > BUFFER_SIZE):
                    newV[0] = currAvG[0];
                    newV[1] = currAvG[1];
                    beliefAge = beliefAge - 2;
            else:
                beliefAge = beliefAge + 1;
                    
    #update the buffer       
    if(side == 0):
        globalL[index][0] = newV[0];
        globalL[index][1] = newV[1]; 
        frameNum = frameNum + 1 #avoid the double incrementation
    else:
        globalR[index][0] = newV[0];
        globalR[index][1] = newV[1];

#apply inter-frame smoothing        
def get_Smoothened(side):
    global globalR
    global globalL 
    global frameNum
    global BUFFER_SIZE
    success = False
    #set_trace();
    limit = min(frameNum-1, BUFFER_SIZE)
    slopeL = 0; 
    interceptL = 0; 
    result = [0, 0];
    #set_trace();
    if limit > 0:
        for i in range(0, limit):
            if(side == 0):
                slopeL = slopeL + globalL[i][0];
                interceptL = interceptL + globalL[i][1];
            else: 
                slopeL = slopeL + globalR[i][0];
                interceptL = interceptL + globalR[i][1];   
                
        slopeL = slopeL / limit; 
        interceptL = interceptL / limit; 
    if(abs(slopeL) > 0):
        result = [slopeL, interceptL];
        success = True;
    else:
        success = False;
    return {'success' : success, 'smoothened' : result};
    
#fetch the last output information     
def get_last_frame(side):
    global frameNum 
    global BUFFER_SIZE
    global globalR
    global globalL
    success = False; 
    retValue = [0, 0]
    isHistoryAvailable = frameNum > 1
    if(isHistoryAvailable):
        index = (frameNum-1)% BUFFER_SIZE
        if(side == 1):
            retValue = globalR[index];
        else:
            retValue = globalL[index];
        success = True; 
    return{'success':success, 'past':retValue};

#select the nearest neighbour to the last output 
def get_best_twin(side, lines):
    #global beliefAge
    bestDistance = 100000;  
    retValue = [0,0];
    success = False;
    past = get_last_frame(side);
    areLinesNotNull = (lines is not None);
    if(past['success']) & areLinesNotNull:
        currAvG = past['past'];
        for line in lines:
            for x1,y1,x2,y2 in line:
                coeffs = np.polyfit([x1, x2], [y1, y2], 1);
                slopeMagOk = (abs(coeffs[0]) > 0.2)
                slopeDirOk = ((coeffs[0] * currAvG[0]) > 0)
                #fitOk = check_Fit(coeffs, currAvG);
                #beliefOk = (beliefAge > 10)
                if(slopeDirOk & slopeMagOk):
                    distance = pow((coeffs[0] - currAvG[0]), 2) + pow((coeffs[1] - currAvG[1]), 2);
                    if(distance < bestDistance):
                        bestDistance = distance; 
                        retValue[0] = coeffs[0];
                        retValue[1] = coeffs[1];
                        success = True;
    return {'success':success, 'twin':retValue }

#distinguish left and right edges. 
#exclude too horizontal edges 
#average the rest of the edges as the output     
def get_averaged_line(side, lines):
    success = False;
    retValue = [0, 0];
    numLines = 0; 
    if(lines is not None):
        for line in lines:
            for x1,y1,x2,y2 in line:
                coeffs = np.polyfit([x1, x2], [y1, y2], 1);
                absSlope = abs(coeffs[0]);
                if(absSlope > 0.2):
                    ySpacing = abs(y1 - y2)
                    if(side == 1):
                        correctSide = (coeffs[0] > 0);
                    else:
                        correctSide = (coeffs[0] < 0);
                    if (correctSide) & (ySpacing > 5) :
                        retValue[0] = retValue[0] + coeffs[0]; 
                        retValue[1] = retValue[1] + coeffs[1];
                        numLines = numLines + 1;
    if numLines > 1:
        retValue[0] = retValue[0]/numLines;
        retValue[1] = retValue[1]/numLines;
        success = True;
    return {'success':success, 'average': retValue}

#extend a line from the base of the image to the upper ROI boundary
def extrapolate_line(img, line):
    yDown = img.shape[0];
    yUp = yDown/2 + (65/540 * img.shape[0]);
    if(abs(line[0]) > 0):
        xDown = (yDown - line[1])/line[0]
        xUp   = (yUp   - line[1])/line[0]
    else: 
        xDown = 0; 
        xUp = 0; 
    retValue = [0, 0, 0, 0];
    retValue[0] = int(xUp);
    retValue[1] = int(yUp);
    retValue[2] = int(xDown);
    retValue[3] = int(yDown);
    return retValue; 
    
def merge_lines(lines, side):
    global frameNum
    global beliefAge
    success = False
    retValue = [0, 0]
    averagingResult = get_averaged_line(side, lines)
    projectionResult = get_best_twin(side, lines)
    isHistoryAvailable = (frameNum > 1)
    
    #assing current value
    if(projectionResult['success']):
        currValue = projectionResult['twin'];
        retValue = [currValue[0], currValue[1]];
        success = True;
    else:
        if(averagingResult['success']):
            currValue = averagingResult['average'];
            retValue = [currValue[0], currValue[1]];
            success = True;
    if(success):
        update_Buffer(currValue, side);
    else:
        beliefAge = beliefAge - 2;
        
    #interframe smoothing
    if (isHistoryAvailable):
        smooth = get_Smoothened(side);
        if(smooth['success']):
            smoothLine = smooth['smoothened'];
            retValue = [smoothLine[0], smoothLine[1]];
            success = True
    return {'success' : success, 'merged' : retValue}  
    
def intersect_at_given_distance(line1, line2, d_req):
    cross_trial = calculate_intersection(line1, line2);
    success = False;
    retValue = [0, 0, 0, 0];
    if(cross_trial['success']):
        cross = cross_trial['result'];
        coeffs1 = np.polyfit([line1[0], line1[2]], [line1[1], line1[3]], 1);
        coeffs2 = np.polyfit([line2[0], line2[2]], [line2[1], line2[3]], 1);
        y_0 = cross[1];
        y_100 = y_0 + 100; 
        x1_100 = (y_100 - coeffs1[1])/coeffs1[0];
        x2_100 = (y_100 - coeffs2[1])/coeffs2[0];
        d_100 = (x1_100 - x2_100);
    
        ydcoeffs = np.polyfit([y_0, y_100], [0, d_100], 1);
        y_req = (d_req - ydcoeffs[1])/ydcoeffs[0];
        x1_req = (y_req - coeffs1[1])/coeffs1[0];
        x2_req = (y_req - coeffs2[1])/coeffs2[0];
        success = True
        retValue = [int(x1_req), int(y_req), int(x2_req), int(y_req)]; 
    return {'success': success, 'result': retValue};
    
def calculate_intersection( line1, line2 ):
    Ax = line1[0]; 
    Ay = line1[1];
    Bx = line1[2];
    By = line1[3];
    Cx = line2[0];
    Cy = line2[1];
    Dx = line2[2];
    Dy = line2[3];
    success = False; 
    retValue = [0, 0];
    
    magAB = math.sqrt(pow((Bx - Ax), 2) + pow((By - Ay), 2));
    magCD = math.sqrt(pow((Dx - Cx), 2) + pow((Dy - Cy), 2));
    if((magAB > 0) & (magCD > 0)):
        ux = (Bx - Ax)/magAB;
        uy = (By - Ay)/magAB;
        vx = (Dx - Cx)/magCD;
        vy = (Dy - Cy)/magCD;
        Px = Cx - Ax; 
        Py = Cy - Ay; 

        s = (Py * ux - uy*Px)/(uy*vx - vy*ux);
        x = Cx + s*vx; 
        y = Cy + s*vy;
        retValue = ([int(x), int(y)])
        success = True
    return {'success': success, 'result': retValue}
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
            
    LEFT = 0; 
    RIGHT = 1; 
    leftFound = False; 
    rightFound = True; 
    GiveATryR = merge_lines(lines, RIGHT)
    lineR = [0, 0, 0, 0];
    lineL = [0, 0, 0, 0];
    if(GiveATryR['success']):
        mergedLineR = GiveATryR['merged'];
        lineR = extrapolate_line(img, mergedLineR);
        rightFound = True; 
        
    GiveATryL = merge_lines(lines, LEFT)
    if(GiveATryL['success']):
        mergedLineL = GiveATryL['merged'];
        lineL = extrapolate_line(img, mergedLineL);
        leftFound = True; 
        
    if(leftFound & rightFound): 
        cross_trial = intersect_at_given_distance(lineR, lineL, 50);
        if(cross_trial['success']):
            cross = cross_trial['result'];
            cv2.line(img, (cross[0], cross[1]), (lineR[2], lineR[3]), color, thickness);
            cv2.line(img, (cross[2], cross[3]), (lineL[2], lineL[3]), color, thickness);
    else:
        if(leftFound):
            cv2.line(img, (lineL[0], lineL[1]), (lineL[2], lineL[3]), color, thickness); 
        else:
            if(rightFound):
                cv2.line(img, (lineR[0], lineR[1]), (lineR[2], lineR[3]), color, thickness);
        
def init():
    global frameNum 
    global beliefAge 
    frameNum = 0; 
    beliefAge = 0; 

def get_ROI(image):
    #Clip the ROI 
    xLength = image.shape[1];
    yLength = image.shape[0];
    
    shiftUp = 75/540 * yLength; 
    shiftSideUp = 400/960 *xLength;
    BoundaryUp = image.shape[0]/2 +shiftUp;
    BoundaryDown = yLength;
    BoundaryUpLX = shiftSideUp; 
    BoundaryUpRX = xLength - shiftSideUp; 
    LeftUp = [BoundaryUpLX, BoundaryUp];
    LeftDown = [0, BoundaryDown];
    RightUp = [BoundaryUpRX, BoundaryUp];
    RightDown = [xLength-0, BoundaryDown];
 
    BoundaryL = np.polyfit([LeftDown [0], LeftUp [0]], [LeftDown [1], LeftUp [1]], 1);
    BoundaryR = np.polyfit([RightDown[0], RightUp[0]], [RightDown[1], RightUp[1]], 1);

    XX, YY = np.meshgrid(np.arange(0, image.shape[1]),                          np.arange(0, image.shape[0]))
    GoodIndeces = (YY <= BoundaryUp)                | (YY >= BoundaryDown)                 | (YY <= (XX * BoundaryL[0] + BoundaryL[1])) 
    return GoodIndeces

def execute_pipeline(image):
    grayIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
        
    #Clip the ROI 
    GoodIndeces = get_ROI(image);

    #Apply a Gaussian Kernel 
    GaussianWindow = 7; 
    GaussianSmoothed = cv2.GaussianBlur(grayIm, (GaussianWindow, GaussianWindow), 0)

    #Canny Edge Detector 
    CannyThesholdLow = 50;
    CannyThresholdHigh = 150; 
    Edges = cv2.Canny(GaussianSmoothed, CannyThesholdLow, CannyThresholdHigh)
    Edges[GoodIndeces]=0

    # Apply the hough transform 
    rho = 1 
    theta = np.pi/180 
    threshold = 14     
    min_line_len = 20 
    max_line_gap = 5    
    houghLines = hough_lines(Edges, rho, theta, threshold, min_line_len, max_line_gap);
    
    result = weighted_img(houghLines, image, 0.5, 0.5, 0.)
    return result 

def play_save_video(inputPath, outputPath):
    init() 
    clip1 = VideoFileClip(inputPath);
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    get_ipython().run_line_magic('time', 'white_clip.write_videofile(outputPath, audio=False)')          


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[6]:


os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[7]:



basePath = "test_images/"; 
outputFolder = "test_images_output/";
myDir = os.listdir(basePath)
for s in myDir:
    currInputPath = basePath + s; 
    currOutputPath = outputFolder + s;
    image = mpimg.imread(currInputPath);
    overlayImg = execute_pipeline(image);
    print(currOutputPath)
    mpimg.imsave(currOutputPath, overlayImg)


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[8]:



def process_image(image):
    result = execute_pipeline(image)
    return result


# In[9]:



#basePath = "test_images/"; 
#outputFolder = "test_images_output/";
#myDir = os.listdir(basePath)
#for s in myDir:
#    currInputPath = basePath + s; 
#    currOutputPath = outputFolder + s;
#    image = mpimg.imread(currInputPath);
#    result = process_image(image)
#    print(currOutputPath)
#    mpimg.imsave(currOutputPath, result)


# Let's try the one with the solid white lane on the right first ...

# In[10]:


white_output = 'test_videos_output/solidWhiteRight.mp4'

play_save_video("test_videos/solidWhiteRight.mp4", "test_videos_output/solidWhiteRight.mp4")


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[11]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[12]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'

play_save_video("test_videos/solidYellowLeft.mp4", "test_videos_output/solidYellowLeft.mp4")


# In[13]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[14]:


challenge_output = 'test_videos_output/challenge.mp4'
play_save_video("test_videos/challenge.mp4", "test_videos_output/challenge.mp4")


# In[15]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# # IMPORTANT! 
# If you are interested to see extra examples for the performance of this algorithm, you can check it [Here](https://www.youtube.com/playlist?list=PLzCBDDtp2hMdCsoM-LQYECuMvt1MW6CJ8)
