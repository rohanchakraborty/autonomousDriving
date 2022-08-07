import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image_lane, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,[5,5],0)
    canny = cv2.Canny(blur,50,150)
    return canny

def regionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

image = cv2.imread('test_image.jpg') # read image from the file
image_lane = np.copy(image)
canny = canny(image_lane) # 1:3 ratio
croppedImage = regionOfInterest(canny)
#plt.imshow(canny)
#plt.show() # shows the image infinitely till a key is pressed
cv2.imshow("result",croppedImage)
cv2.waitKey(0)
