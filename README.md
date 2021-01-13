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
![image](https://user-images.githubusercontent.com/72300138/104424343-1beb5500-5534-11eb-9b10-a95e6b93e8f5.png)


2)Develop a program to perform linear tranformation on an image:Scaling and rotation
#SCALING
import cv2 
imgclr=cv2.imread("imgred.jpg") 
res = cv2.resize(imgclr,(300,300),interpolation=cv2.INTER_CUBIC) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()
Output:
![image](https://user-images.githubusercontent.com/72300138/104424627-7ab0ce80-5534-11eb-8eb6-6e763a34dadc.png)

#ROTATION
import cv2 
imgclr=cv2.imread("colorimg.jpg") 
(row, col) = imgclr.shape[:2] 
M = cv2.getRotationMatrix2D((col / 2, row/ 2), 45, 1)
res = cv2.warpAffine(imgclr, M, (col,row)) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()
Output:
![image](https://user-images.githubusercontent.com/72300138/104425220-4689dd80-5535-11eb-95d9-e64bd5e2f68a.png)

3)#Develop a program to find the sum and mean of a set of images.
#a.Create  'n' number of images abd read them from the directory and perform the operations.
sum and mean
import cv2
import glob 
import numpy as np
from PIL import Image
path=glob.glob("E:\pics\*.jpg")
for file in path:
    print(file)
    image=cv2.imread(file)
    sum=image+sum
mean=sum/20
cv2.imshow("Sum",sum)
cv2.waitKey(0)
cv2.imshow("Mean",mean)
cv2.waitKey(0)
cv2.destroyAllWindows()
Output:
![image](https://user-images.githubusercontent.com/72300138/104425703-d62f8c00-5535-11eb-94ce-498b71c830a5.png)

#4)Develop a program to convert the color image to grayscale and binary image
import cv2
image=cv2.imread("pic6.jpg")
cv2.imshow("Original Image",image)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray)
cv2.waitKey(0)
ret,binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary Image",binary)
cv2.waitKey(0)
cv2.destroyAllWindows()




#5)Develop a program to convert the color images to different color spaces
import cv2
image=cv2.imread("pic6.jpg")
cv2.imshow("Original Image",image)
cv2.waitKey(0)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray)
cv2.waitKey(0)
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
cv2.imshow(" YCrcb",ycrcb)
cv2.waitKey(0)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow(" HSV",hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
Output:
![image](https://user-images.githubusercontent.com/72300138/104426626-09bee600-5537-11eb-820f-f6aad3381e1b.png)



#6)Develop a program to create an image from 2D array(generate an array of random size)
)creating 2d image
import numpy as np 
import cv2
from PIL import Image as im 
w,h=20,230
i = np.zeros((h,w,3), dtype=np.uint8)
i[0:256, 0:256]=[123,20,35]
data = im.fromarray(i, 'RGB') 
data.save('image.jpg') 
data.show()
Output:
![image](https://user-images.githubusercontent.com/72300138/104426484-da0fde00-5536-11eb-804b-8bbdaf3d3cd1.png)







