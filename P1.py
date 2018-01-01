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
#-----------------------------------------------------------------------------------------------    

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
    areLinesNotNull = (lines is not None);
    if(past['success']) & areLinesNotNull:
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
    
def merge_lines(lines, side):
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
        cross = intersect_at_given_distance(lineR, lineL, 50);
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
    
def play_save_video(inputPath, outputPath):
    init() 
    clip1 = VideoFileClip(inputPath);
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    get_ipython().run_line_magic('time', 'white_clip.write_videofile(outputPath, audio=False)')
          
    
import os
os.listdir("test_images/")

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


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
    
def process_image(image):    
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
    
    result = weighted_img(houghLines, image, 1., 1., 0.)
    return result


#newSample3_output = 'test_videos_output/newSample3.mp4'
#play_save_video("test_videos/newSample3.mp4", "test_videos_output/newSample3.mp4")
    
#newSample3_output = 'test_videos_output/sample8.mp4'
#play_save_video("test_videos/sample8.mp4", "test_videos_output/sample8.mp4")
#newSample3_output = 'test_videos_output/sample9.mp4'
#play_save_video("test_videos/sample9.mp4", "test_videos_output/sample9.mp4") 
 
#white_output = 'test_videos_output/solidWhiteRight.mp4'
#play_save_video("test_videos/solidWhiteRight.mp4", "test_videos_output/solidWhiteRight.mp4")

#yellow_output = 'test_videos_output/solidYellowLeft.mp4'
#play_save_video("test_videos/solidYellowLeft.mp4", "test_videos_output/solidYellowLeft.mp4")

challenge_output = 'test_videos_output/challenge.mp4'
play_save_video("test_videos/challenge.mp4", "test_videos_output/challenge.mp4")


#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(white_output))

#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(yellow_output))

#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(challenge_output))

