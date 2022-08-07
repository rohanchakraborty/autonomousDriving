# Edge Detection : Identifying sharp changes in intensity
#                  in the adjacent pixels

# Gradient : Measure of change in brightness over adjacent pixels

import cv2
#Step 1, convert to grayscale -- i.e. one channel
#        processing a single channel is easier that 3 channel
# Step 2, apply gaussian blur - smoothen image
import numpy as np

image_lane = np.copy(cv2.imread('test_image.jpg'))
gray = cv2.cvtColor(image_lane,cv2.COLOR_RGB2GRAY)


# Step 2, apply gaussian blur - smoothen image
# filter out noise (chances to detect false array : reduced)

# weighted average set to each pixel via
blur = cv2.GaussianBlur(gray,[5,5],0)
cv2.imshow("result",gray)
cv2.waitKey(0)
