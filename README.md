# image-processing-
Assignment 1
#Develop a program to display grayscale image using read and write operation
#imread() function is used to read the file.
#imwrite() function is used to save file in memory disk
import cv2
imgclr=cv2.imread("imgred.jpg",1)
imggry = cv2.cvtColor(imgclr, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',imggry)
cv2.imwrite('grayimg.jpg',imggry)
cv2.waitKey()
Output:
