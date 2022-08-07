import cv2

image = cv2.imread('test_image.jpg') # read image from the file
cv2.imshow('result',image)
cv2.waitKey(0)
