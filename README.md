## image-processing-
Assignment 1
## 1)Develop a program to display grayscale image using read and write operation
#import cv2 is used to import opencv.imshow() is for display.waitkey(n) is to wait for n miliseconds.When n=0,execution is paused until a key is pressed.
#destroyAllWindow() function closes all the window
#imread() function is used to read the file.
#imwrite() function is used to save file in memory disk
#COLOR_BGR2GRAY will covert color images into grayscale image
import cv2
imgclr=cv2.imread("imgred.jpg",1)
imggry = cv2.cvtColor(imgclr, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',imggry)
cv2.imwrite('grayimg.jpg',imggry)
cv2.waitKey()
Output:



![image](https://user-images.githubusercontent.com/72300138/104424343-1beb5500-5534-11eb-9b10-a95e6b93e8f5.png)


## 2)Develop a program to perform linear tranformation on an image:Scaling and rotation
#SCALING:In computer graphics and digital imaging, image scaling refers to the resizing of a digital image.Here we used resize function for scaling.
import cv2 
imgclr=cv2.imread("imgred.jpg") 
res = cv2.resize(imgclr,(300,300),interpolation=cv2.INTER_CUBIC) 
cv2.imshow('imagecolor',imgclr)
cv2.imshow('imagecolor1',res)
cv2.waitKey()
Output:



![image](https://user-images.githubusercontent.com/72300138/104424627-7ab0ce80-5534-11eb-8eb6-6e763a34dadc.png)

#ROTATION:Rotate Image in Python using OpenCV To rotate an image, apply a matrix transformation. To create a matrix transformation, use the cv2.getRotationMatrix2D () method and pass the origin that we want the rotation to happen around. If we pass the origin (0, 0), then it will start transforming the matrix from the top-left corner.
warpAffine() function mainly uses the transformation matrix to transform the images such as rotation affine,translation etc.
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

## 3)#Develop a program to find the sum and mean of a set of images.
#a.Create  'n' number of images and read them from the directory and perform the operations.
#The glob() function returns an array of filenames or directories matching a specified pattern.
#sum-Adding all images.mean-Finding average of all the images.
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

## 4)Develop a program to convert the color image to grayscale and binary image
#COLOR_BGR2GRAY will convert color images into grayscale image.
#Simple Thresholding The basic Thresholding technique is Binary Thresholding. For every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise, it is set to a maximum value.
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
Output:



![image](https://user-images.githubusercontent.com/72300138/104426911-6e7a4080-5537-11eb-9dec-b62fe7473e23.png)




## 5)Develop a program to convert the color images to different color spaces
#COLOR_BGR2GRAY will covert color images into grayscale image.
#COLOR_BGR2YCrCb-Y represents Luminance or Luma component, Cb and Cr are Chroma components. Cb represents the blue-difference (difference of blue component and Luma Component). Cr represents the red-difference (difference of red component and Luma Component).
#COLOR_BGR2HSV-H : Hue represents dominant wavelength.
S : Saturation represents shades of color.
V : Value represents Intensity.
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



## 6)Develop a program to create an image from 2D array(generate an array of random size)
#creating 2d image
#Numpy zeros np.zeros () function in python is used to get an array of given shape and type filled with zeros. You can pass three parameters inside function np.zeros shape, dtype and order. Numpy zeros function returns an array of the given shape
#im.fromarray converts array into image of height h,width w.
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



## 7)Develop a program to find the sum of neighbour of each elements in the matrix
#The numpy.zeros() function returns a new array of given shape and type, with zeros.append () Syntax: list_name.append (‘value’) It takes only one argument. This function appends the incoming element to the end of the list as a single new element.

import numpy as np
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2):
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError:
                pass
    return sum(l)-M[x][y] 
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)
Output:
Original matrix:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
Summed neighbors matrix:
 [[11. 19. 13.]
 [23. 40. 27.]
 [17. 31. 19.]]
 
 ## 8)Write a c++ program to perform operator overloading 
 #include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];

   }
  }
 
 
 };
 void operator+(matrix a1)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
   
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }

 };

  void operator-(matrix a2)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }
   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };

 void operator*(matrix a3)
 {
  int c[i][j];

  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
 };

};

int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}
Output:
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
9
8
7
6
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
5
4
3
2
addition is
 14      12
 10      8
subtraction is
 4      4
 4      4
multiplication is
 69     52
 53     40


## Develop a program to implement negative tranformation of an image
#imread() function is to read image.imshow() is for display.waitkey(n) is to wait for n miliseconds.When n=0,execution is paused until a key is pressed.
#destroyAllWindow() function closes all the windows.import cv2 is used to import opencv.
#The negative transformation is given by s=L-1-r.Where L=Maximum value(256),r=image pixel.
import cv2 
img = cv2.imread('pic1.jpeg') 
cv2.imshow("Original",img)
cv2.waitKey(0)
neg=255-img
cv2.imshow("negetive",neg)
cv2.waitKey(0);
cv2.destroyAllWindows()
Output:



![image](https://user-images.githubusercontent.com/72300138/105324512-6007d680-5b80-11eb-8ccd-a1fac8df42f8.png)
![image](https://user-images.githubusercontent.com/72300138/105324581-74e46a00-5b80-11eb-9855-964a09504a6b.png)

#Contrast of an image
#Python Imaging Library (PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.To load an image from a file, use the open() function in the Image module.Image Enhancer uses contrast enhancement techniques to optimize the photo color. show() function is to display.
from PIL import Image, ImageEnhance
img = Image.open("pic1.jpeg")
img.show()
img=ImageEnhance.Color(img)
img.enhance(2.0).show()
Output:

![image](https://user-images.githubusercontent.com/72300138/105326568-baa23200-5b82-11eb-8c01-3bef31a17cc2.png)
![image](https://user-images.githubusercontent.com/72300138/105326657-d4dc1000-5b82-11eb-8e03-9d7a985d9915.png)










