# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:09:27 2017

@author: Gavin
"""

import numpy as np
import cv2
import glob

def auto_canny(img, sigma = 0.33):
    med = np.median(img)
    
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edged = cv2.Canny(img, lower, upper)
    
    return edged

def prep_image(gray):
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9,9),0)
    tresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)[1]
    return tresh

for img in glob.glob("Serie2/*.jpg"):
    img = cv2.imread(img)
    resImg = cv2.resize(img,None, fx = .15, fy = .15)
    resImg = resImg[:,:,2]
    prepImg = prep_image(resImg)
    #autoImg = auto_canny(prepImg)
    cv2.imshow("Original", resImg)
    cv2.imshow("Edges", prepImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()