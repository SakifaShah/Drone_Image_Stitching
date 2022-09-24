# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:39:58 2022

@author: Eutech
"""

#import the necessary packages
import os
import cv2
import math
import numpy as np
import imutils
image_paths=['file_1.JPG',
             'file_2.JPG',
             'file_3.JPG',
             'file_4.JPG',
             'file_5.JPG',
             'file_6.JPG',
             'file_7.JPG',
             'file_8.JPG',
             'file_9.JPG',
             'file_10.JPG',
             'file_11.JPG',
             'file_12.JPG',
             'file_13.JPG',
             'file_14.JPG',
             'file_15.JPG',
             'file_16.JPG',
             'file_17.JPG',
             'file_18.JPG',
             'file_19.JPG',
             'file_20.JPG',
             'file_21.JPG',
             'file_22.JPG',
             'file_23.JPG',
             'file_24.JPG',
             'file_25.JPG',
             'file_26.JPG',
             'file_27.JPG',
             'file_28.JPG',
             'file_29.JPG',
             'file_30.JPG',
             'file_31.JPG',
             'file_32.JPG',
             'file_33.JPG',
             'file_34.JPG',
             'file_35.JPG',
             'file_36.JPG',
             'file_37.JPG',
             'file_38.JPG',
             'file_39.JPG',
             'file_40.JPG']


imgs = []
for i in image_paths:
    img = cv2.imread(i)
    #img = cv2.resize(img, (1500,1500), interpolation=cv2.INTER_AREA)
    imgs.append(img)
    print(img.shape)
    
stitchy=cv2.Stitcher_create()
(dummy,output)=stitchy.stitch(imgs)

if dummy != cv2.STITCHER_OK:
  # checking if the stitching procedure is successful
  # .stitch() function returns a true value if stitching is
  # done successfully
    print("stitching ain't successful")
else:
    print('Your Panorama is ready!!!')
 
# final output
#cv2.imshow('final result',output)
cv2.imwrite("StitchedOutput", output)


# =============================================================================
# if not dummy:
#    
#     output = cv2.copyMakeBorder(output, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
# 
#     gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
#     thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
# 
#     #cv2.imshow("Threshold Image", thresh_img)
#     #cv2.waitKey(0)
# 
#     contours, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
#     #contours = imutils.grab_contours(contours)
#     areaOI = max(contours, key=cv2.contourArea)
# 
#     mask = np.zeros(thresh_img.shape, dtype="uint8")
#     x, y, w, h = cv2.boundingRect(areaOI)
#     cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
# 
#     minRectangle = mask.copy()
#     sub = mask.copy()
# 
#     while cv2.countNonZero(sub) > 0:
#         minRectangle = cv2.erode(minRectangle, None)
#         sub = cv2.subtract(minRectangle, thresh_img)
# 
# 
#     contours, hierarchy = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
#     contours = imutils.grab_contours(contours)
#     print(len(contours))
# =============================================================================
    #areaOI = max(contours, key=cv2.contourArea)

    #cv2.imshow("minRectangle Image", minRectangle)
    #cv2.waitKey(0)

    #x, y, w, h = cv2.boundingRect(areaOI)

    #stitched_img = stitched_img[y:y + h, x:x + w]

    #cv2.imwrite("stitchedOutputProcessed2.png", output)

    #cv2.imshow("Stitched Image Processed", stitched_img)

    #cv2.waitKey(0)



# =============================================================================
# else:
#     print("Images could not be stitched!")
#     print("Likely not enough keypoints being detected!")
# =============================================================================
