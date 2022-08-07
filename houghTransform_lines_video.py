# 2D coordinate x,y, line = mx+b
#Line Detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,[5,5],0)
    canny = cv2.Canny(blur,50,150)
    return canny

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def averageSlopeIntercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def regionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def displayLines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

#image = cv2.imread('test_image.jpg') # read image from the file
#image_lane = np.copy(image)
#canny_image = canny(image_lane) # 1:3 ratio
#croppedImage = regionOfInterest(canny_image)
#lines = cv2.HoughLinesP(croppedImage,2,np.pi/180,100,np.array([]),minLineLength = 40,maxLineGap=5)
#averagedLines = averageSlopeIntercept(image_lane,lines)
#line_image = displayLines(image_lane,averagedLines)
#combiLine = cv2.addWeighted(image_lane,0.8,line_image,1,1)
#convolution_line_image = cv2.addWeighted(image_lane,0.8,averagedLines,1,1)
#plt.imshow(canny)
#plt.show() # shows the image infinitely till a key is pressed
#cv2.imshow("result",combiLine)
#cv2.waitKey(0)
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = regionOfInterest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    averaged_lines = averageSlopeIntercept(frame, lines)
    line_image = displayLines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
