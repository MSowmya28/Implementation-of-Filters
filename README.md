# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the necessary modules.

### Step2
For performing smoothing operation on a image.
Average filter
```
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```
Weighted average filter:
```
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```
Gaussian Blur:
```
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
```
Median filter:
```
median=cv2.medianBlur(image2,13)
```
### Step3
For performing sharpening on a image.

Laplacian Kernel:
```
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```
Laplacian Operator:
```
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```
### Step4
Display all the images with their respective filters.

## Program:
```
Developed By   : M.Sowmya
Register Number:212221230107
```
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread('dip6.png')
img2=cv2.cvtColor (img1, cv2.COLOR_BGR2RGB)
```

### 1. Smoothing Filters

i) Using Averaging Filter
```
kernel1 = np.ones((11,11),np.float32)/121
img3 = cv2.filter2D(img2,-1,kernel1)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img3)
plt.title("Filtered")
plt.axis("off")
```
ii) Using Weighted Averaging Filter
```
kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/25
img3 = cv2.filter2D(img2,-1,kernel2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img3)
plt.title("Filtered")
plt.axis("off")
```
iii) Using Gaussian Filter
```
gaussian_blur = cv2.GaussianBlur(src = img2, ksize = (11,11), sigmaX=0, sigmaY=0)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Filtered")
plt.axis("off")
```

iv) Using Median Filter
```
median = cv2.medianBlur(src=img2,ksize = 11)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Filtered (Median)")
plt.axis("off")
```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```
kernel3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_kernel = cv2.filter2D(img2,-1,kernel3)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_kernel)
plt.title("Filtered (Laplacian Kernel)")
plt.axis("off")
```
ii) Using Laplacian Operator
```
laplacian=cv2.Laplacian(img2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(img2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters


i) Using Averaging Filter:

![output](./dip%20avg.png)


ii) Using Weighted Averaging Filter:

![output](./dip%20avg%20weigth.png)


iii) Using Gaussian Filter:

![ouput](./dip%20guassian.png)


iv) Using Median Filter:

![output](./dip%20median.png)


### 2. Sharpening Filters


i) Using Laplacian Kernal:

![output](./dip%20lap%20ker.png)


ii) Using Laplacian Operator:

![output](./dip%20lap%20op.png)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
