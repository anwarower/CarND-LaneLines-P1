#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

import math
from IPython.core.debugger import set_trace

BUFFER_SIZE = 25; 
globalR = np.zeros((BUFFER_SIZE, 2))
globalL = np.zeros((BUFFER_SIZE, 2))
frameNum = 0
beliefAge = 0

def check_Fit(line1, line2):
    slopeThreshold = 0.1; 
    interceptThreshold = 25; 
    slopeOK     = (abs(line1[0] - line2[0]) < slopeThreshold);
    interceptOK = (abs(line1[1] - line2[1]) < interceptThreshold)
    return (slopeOK & interceptOK);

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
    #helper methods 
    #----------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------------    
    
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

def get_best_twin(side, lines):
    bestDistance = 100000;  
    retValue = [0,0];
    success = False;
    past = get_last_frame(side);
    if(past['success']):
        currAvG = past['past'];
        for line in lines:
            for x1,y1,x2,y2 in line:
                coeffs = np.polyfit([x1, x2], [y1, y2], 1);
                slopeMagOk = (abs(coeffs[0]) > 0.2)
                slopeDirOk = ((coeffs[0] * currAvG[0]) > 0)
                if(slopeDirOk & slopeMagOk):
                    distance = pow((coeffs[0] - currAvG[0]), 2) + pow((coeffs[1] - currAvG[1]), 2);
                    if(distance < bestDistance):
                        bestDistance = distance; 
                        retValue[0] = coeffs[0];
                        retValue[1] = coeffs[1];
                        success = True;
    return {'success':success, 'twin':retValue }

def get_averaged_line(side, lines):
    success = False;
    retValue = [0, 0];
    numLines = 0; 
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

def get_averaged_line2(side, lines):
    success = False;
    retValue = [0, 0];
    numLines = 0;
    global beliefAge
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
                closeEnough = True;    
                if(beliefAge > 25):
                    smooth = get_Smoothened(side);
                    if(smooth['success']):
                        closeEnough = check_Fit(smooth['smoothened'], coeffs);
                if (correctSide) & (closeEnough) :
                    retValue[0] = retValue[0] + coeffs[0]; 
                    retValue[1] = retValue[1] + coeffs[1];
                    numLines = numLines + 1;
    if numLines > 1:
        retValue[0] = retValue[0]/numLines;
        retValue[1] = retValue[1]/numLines;
        success = True;
    return {'success':success, 'average': retValue}

def extrapolate_line(img, line):
    yDown = img.shape[0];
    yUp = yDown/2 + (65/540 * img.shape[0]);
    xDown = (yDown - line[1])/line[0]
    xUp   = (yUp   - line[1])/line[0]
    
    retValue = [0, 0, 0, 0];
    retValue[0] = int(xUp);
    retValue[1] = int(yUp);
    retValue[2] = int(xDown);
    retValue[3] = int(yDown);
    return retValue; 
    
def merge_line2(lines, side):
    global frameNum
    success = False
    retValue = [0, 0]
    averagingResult = get_averaged_line(side, lines)
    projectionResult = get_best_twin(side, lines)
    isHistoryAvailable = (frameNum > 5)
    
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
        
    #interframe smoothing
    if (isHistoryAvailable):
        smooth = get_Smoothened(side);
        if(smooth['success']):
            smoothLine = smooth['smoothened'];
            retValue = [smoothLine[0], smoothLine[1]];
            success = True
    return {'success' : success, 'merged' : retValue}  

def merge_line(lines, side):
    global frameNum 
    global beliefAge
    success = False
    retValue = [0, 0]
    averagingResult = get_averaged_line2(side, lines);
    
    isHistoryAvailable = (frameNum > 5)
    
    if (averagingResult['success']): 
        avg = averagingResult['average'];
        update_Buffer(avg, side)
         
        if(isHistoryAvailable):
            smooth = get_Smoothened(side);
            if(smooth['success']):
                smoothLine = smooth['smoothened'];
                retValue[0] = smoothLine[0];
                retValue[1] = smoothLine[1];
        else:
            retValue[0] = avg[0]; 
            retValue[1] = avg[1]; 
        success = True;
    else:
        beliefAge = beliefAge - 2;
        if (isHistoryAvailable):
            smooth = get_Smoothened(side);
            if(smooth['success']):
                smoothLine = smooth['smoothened'];
                retValue[0] = smoothLine[0];
                retValue[1] = smoothLine[1];
                success = True
    return {'success' : success, 'merged' : retValue}
    
def intersect_at_given_distance(line1, line2, d_req):
    cross = calculate_intersection(line1, line2);
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
    return [int(x1_req), int(y_req), int(x2_req), int(y_req)];
    
import math
def calculate_intersection( line1, line2 ):
    Ax = line1[0]; 
    Ay = line1[1];
    Bx = line1[2];
    By = line1[3];
    Cx = line2[0];
    Cy = line2[1];
    Dx = line2[2];
    Dy = line2[3];
    
    magAB = math.sqrt(pow((Bx - Ax), 2) + pow((By - Ay), 2));
    magCD = math.sqrt(pow((Dx - Cx), 2) + pow((Dy - Cy), 2));
    ux = (Bx - Ax)/magAB;
    uy = (By - Ay)/magAB;
    vx = (Dx - Cx)/magCD;
    vy = (Dy - Cy)/magCD;
    Px = Cx - Ax; 
    Py = Cy - Ay; 

    s = (Py * ux - uy*Px)/(uy*vx - vy*ux);
    x = Cx + s*vx; 
    y = Cy + s*vy;
    return ([int(x), int(y)])
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        coeffs = np.polyfit([x1, x2], [y1, y2], 1);
    #        absSlope = abs(coeffs[0]);
    #        if(absSlope > 0.3):
    #            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                
                      
    LEFT = 0; 
    RIGHT = 1; 
    leftFound = False; 
    rightFound = True; 
    GiveATryR = merge_line2(lines, RIGHT)
    if(GiveATryR['success']):
        mergedLineR = GiveATryR['merged'];
        lineR = extrapolate_line(img, mergedLineR);
        rightFound = True; 
        #cv2.line(img, (lineR[0], lineR[1]), (lineR[2], lineR[3]), color, thickness);
        
    GiveATryL = merge_line2(lines, LEFT)
    if(GiveATryL['success']):
        mergedLineL = GiveATryL['merged'];
        lineL = extrapolate_line(img, mergedLineL);
        leftFound = True; 
        #cv2.line(img, (lineL[0], lineL[1]), (lineL[2], lineL[3]), color, thickness); 
        
        if(leftFound & rightFound): 
            #cross = calculate_intersection(lineR, lineL);
            cross = intersect_at_given_distance(lineR, lineL, 50);
         
            #cv2.line(img, (cross[0], cross[1]) ,(lineR[2], lineR[3]), color, thickness);
            #cv2.line(img, (cross[2], cross[3]) ,(lineL[2], lineL[3]), color, thickness);
            cv2.line(img, (cross[0], cross[1]), (lineR[2], lineR[3]), color, thickness);
            cv2.line(img, (cross[2], cross[3]), (lineL[2], lineL[3]), color, thickness);
    #lineR = [0, 0, 0, 0];
    #lineL = [0, 0, 0, 0];
    
   # coeffsR = average_poly(img, lines, 1); 
   # if(len(coeffsR)> 1):
   #     #set_trace()
   #     draw_poly(img, coeffsR, color, thickness);
        
   #if(average_line(img, lines, lineL, 0)== 1):
   #     cv2.line(img, (lineL[0], lineL[1]), (lineL[2], lineL[3]), color, thickness); 
        
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def average_poly(img, lines, side):
    avGSlope = 0; 
    avIntercept = 0; 
    numLines = 0;
    yDown = img.shape[0];
    yUp = yDown/2 + 50;
    xPts = [];
    yPts = [];
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
                    xPts = xPts + [x1, x2];
                    yPts = yPts + [y1, y2];
                    numLines = numLines + 1; 
                    
    if numLines > 0:
        #set_trace();
        degree = (numLines * 2) - 1;
        degree = min(degree, 1)
        retValue = np.zeros(degree + 1);
        polyFit = np.polyfit(xPts, yPts, degree);
        for i in range(0, degree+1):
            retValue[i] = polyFit[i];
        return retValue
    else: 
        return []
    return []


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


# In[ ]:


def eval_poly(coeffs, x):
    degree = len(coeffs) - 1; 
    y = 0;
    for i in range(0, degree+1):
        y = y + ((pow(x, i)) * coeffs[degree - i]);
    return y;

def draw_poly(img, coeffs, color=[255, 0, 0], thickness=10):
    x1 = 450; 
    while (x1 < img.shape[1]):
        
        #set_trace();
        y1 = eval_poly(coeffs, x1);
        x2 = x1 + 20; 
        y2 = eval_poly(coeffs, x2);
        #y2 = a * pow(x2, 3) + b * pow(x2, 2) + c * pow(x2, 1) + d * pow(x2, 0);
        cv2.line(img, (x1, int(y1)), (x2, int(y2)), color, thickness); 
        x1 = x2; 
        
def init():
    global frameNum 
    global beliefAge 
    frameNum = 0; 
    beliefAge = 0; 
    
def play_save_video(inputPath, outputPath):
    global frameNum
    frameNum = 0
    clip1 = VideoFileClip(inputPath);
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    get_ipython().run_line_magic('time', 'white_clip.write_videofile(outputPath, audio=False)')
    init()       
    


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


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

# In[5]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[6]:



def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
   
    global globalR
    global globalL
    global frameNum
    
    grayIm = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    #grayIm = cv2.equalizeHist(grayIm)
        
    #Clip the ROI 
    xLength = grayIm.shape[1];
    yLength = grayIm.shape[0];
    
    shiftUp = 75/540 * yLength; 
    shiftSideUp = 400/960 *xLength;
    BoundaryUp = grayIm.shape[0]/2 +shiftUp;
    BoundaryDown = yLength;
    BoundaryUpLX = shiftSideUp; 
    BoundaryUpRX = xLength - shiftSideUp; 
    LeftUp = [BoundaryUpLX, BoundaryUp];
    LeftDown = [0, BoundaryDown];
    RightUp = [BoundaryUpRX, BoundaryUp];
    RightDown = [xLength-0, BoundaryDown];
    #BoundaryL = np.polyfit([0, BoundaryUpLX], [BoundaryDown, BoundaryUp], 1); 
    BoundaryL = np.polyfit([LeftDown [0], LeftUp [0]], [LeftDown [1], LeftUp [1]], 1);
    BoundaryR = np.polyfit([RightDown[0], RightUp[0]], [RightDown[1], RightUp[1]], 1);

    XX, YY = np.meshgrid(np.arange(0, grayIm.shape[1]),                          np.arange(0, grayIm.shape[0]))
    GoodIndeces = (YY <= BoundaryUp)                | (YY >= BoundaryDown)                 | (YY <= (XX * BoundaryL[0] + BoundaryL[1]))                | (YY <= (XX * BoundaryR[0] + BoundaryR[1]))


    #Apply a Gaussian Kernel 
    GaussianWindow = 7; 
    GaussianSmoothed = cv2.GaussianBlur(grayIm, (GaussianWindow, GaussianWindow), 0)

    #Canny Edge Detector 
    CannyThesholdLow = 50;
    CannyThresholdHigh = 150; 
    Edges = cv2.Canny(GaussianSmoothed, CannyThesholdLow, CannyThresholdHigh)
    Edges[GoodIndeces]=0

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 14     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    houghLines = hough_lines(Edges, rho, theta, threshold, min_line_len, max_line_gap);
    
    result = weighted_img(houghLines, image, 1., 1., 0.)
    return result


newSample3_output = 'test_videos_output/newSample3.mp4'
play_save_video("test_videos/newSample3.mp4", "test_videos_output/newSample3.mp4")
    
white_output = 'test_videos_output/solidWhiteRight.mp4'
play_save_video("test_videos/solidWhiteRight.mp4", "test_videos_output/solidWhiteRight.mp4")

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
play_save_video("test_videos/solidYellowLeft.mp4", "test_videos_output/solidYellowLeft.mp4")

challenge_output = 'test_videos_output/challenge.mp4'
play_save_video("test_videos/challenge.mp4", "test_videos_output/challenge.mp4")

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

