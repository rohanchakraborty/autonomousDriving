import cv2
import numpy as np

image = cv2.imread('test_image.jpg') # read image from the file
image_lane = np.copy(image)
gray = cv2.cvtColor(image_lane, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray,[5,5],0)
canny = cv2.Canny(blur,50,150) # 1:3 ratio
cv2.imshow('result',canny)
cv2.waitKey(0) # shows the image infinitely till a key is pressed
