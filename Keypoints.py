# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:34:40 2022

@author: Eutech
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


#load the images
img_1 = cv2.imread('output34.jpg')
img_1 = cv2.resize(img_1, (6593,5351))
plt.imshow(img_1)
plt.show()

img_2 = cv2.imread('output56.jpg')
#plt.imshow(img_2)
#plt.show()

#img_3 = cv2.imread('EP-11-29590_0007_0005.JPG')
#plt.imshow(img_2)
#plt.show()

#convert hte image to gray scale
gray_img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_img_1)
#plt.show()

gray_img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_img_2)
#plt.show()

#gray_img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_img_2)
#plt.show()

#compute the sift keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create() 
kp1, des1 = sift.detectAndCompute(gray_img_1, None)
kp2, des2 = sift.detectAndCompute(gray_img_2, None)
#kp3, des3 = sift.detectAndCompute(gray_img_3, None)

#print(kp1)
#print(type(des2))

#find top matches of descriptors of 2 images
bf = cv2.BFMatcher()
#print(type(bf))

matches = bf.knnMatch(des1, des2, k=2)
#img3 = cv2.drawMatchesKnn(img_1,kp1,img_2,kp2,matches,None)
#plt.imshow(img3)
#plt.show()

#print(matches)
#matches2 = bf.knnMatch(des2, des3, k=2)
#matches3 = bf.knnMatch(des1, des3, k=2)
#matches = matches1 + matches2 + matches3
print(type(matches))
good = []
for m in matches:
    if(m[0].distance<0.5*m[1].distance):
        good.append(m)
matches = np.asarray(good)

#print(good)

#align two images using homography transformation
if(len(matches[:,0])>=4):
    src = np.float32([kp1[m.queryIdx].pt
                      for m in matches[:,0]]).reshape(-1,1,2)
    
    dst = np.float32([kp2[m.trainIdx].pt
                      for m in matches[:,0]]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    print(type(H))
    print(type(masked))
    
else: 
    raise AssertionError('Cant find enough keypoints')
    
#stitch the image
dst = cv2.warpPerspective(img_1, H, ((img_1.shape[0]+img_1.shape[1], img_1.shape[0]+img_1.shape[1])))
plt.imshow(dst)

#warped image
dst[0:img_1.shape[0], 0:img_2.shape[1]] = img_2

#stitched image

cv2.imwrite('output_.jpg', dst)
plt.imshow(dst)
plt.show()


# =============================================================================
# crop = crop_with_argwhere(dst)
#        
# cv2.imwrite('crop_output3_4.jpg', crop)
# plt.imshow(crop)
# plt.show()
# =============================================================================
