# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:40:33 2017

@author: Gavin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:10:59 2017

@author: Gavin
"""

import cv2
import glob
import numpy as np
kNearest = cv2.ml.KNearest_create()


#yellow thresholds
YELLOW_MIN = np.array([10, 50, 50],np.uint8)
YELLOW_MAX = np.array([30,255,255],np.uint8)

SHOW_STEPS = 0

RESIZED_CHAR_IMAGE_WIDTH = 40
RESIZED_CHAR_IMAGE_HEIGHT = 60

def prepImg(img,itr):
    #blurred = cv2.GaussianBlur(img, (1,1),0)
    kernel = np.ones((3,3),np.uint8)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #yellow_tresh = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
    

    #yellow thresholds
    if (itr == 1):     
        YELLOW_MIN = np.array([10, 50, 50],np.uint8)
        YELLOW_MAX = np.array([30,255,255],np.uint8)
        mask = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
        yellow_tresh = cv2.bitwise_and(img,img,mask=mask)
        yellow_tresh = cv2.erode(yellow_tresh,kernel,iterations = 10)
        yellow_tresh = cv2.cvtColor(yellow_tresh,cv2.COLOR_BGR2GRAY)
    elif (itr == 2):
        YELLOW_MIN = np.array([14, 50, 50],np.uint8)
        YELLOW_MAX = np.array([255,255,255],np.uint8) 
        mask = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
        yellow_tresh = cv2.bitwise_and(img,img,mask=mask)
        #yellow_tresh = cv2.GaussianBlur(yellow_tresh, (5,3),0)
        yellow_tresh = cv2.erode(yellow_tresh,kernel,iterations = 6)
        yellow_tresh = cv2.cvtColor(yellow_tresh,cv2.COLOR_BGR2GRAY)

    
    if (SHOW_STEPS == 1):
        cv2.imshow("yellow", yellow_tresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return yellow_tresh

    
def biggestCont(img):
    #find contours
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    biggestArea = 0
    #find biggest contour
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > biggestArea:
            theOne = i
            biggestArea = area
    return contours, theOne


def makeContPretty(cont):
    #makes a bounding rectangle
    rect = cv2.minAreaRect(cont)    
#    size = rect[1]
#    print("size = :",size)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect,box

def deskewAndCrop(img, rect, box):  #copied from the 'net but only way to do it
    W = rect[1][0]+25
    H = rect[1][1]+40
#    W = rect[1][0]
#    H = rect[1][1]
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    
    angle = rect[2]
    if angle < -45:
        angle += 90
    
    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1+25, y2-y1+40)#-50/50
    #size = (x2-x1, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW),int(croppedH)), (size[0]/2, size[1]/2))
    return croppedRotated


def showStuff(cnt, img, prep_img, roi):
    cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
    cv2.imshow("Original", res_img)
    #cv2.imshow("Edges", prep_img)
    cv2.imshow("Crop", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def showTest(img, cnt):
    
    if cnt != None:
        cv2.drawContours(img, [cnt], -1, (0,255,0), 3) 
    cv2.imshow("Testimg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hullCont(cnt):
    hull = cv2.convexHull(cnt)
    return hull

def nothing(c):
    pass

def frptConts(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
    return approx

####################################################################################
def secondTresh(img):
    
    
    
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_img = cv2.GaussianBlur(hsv_img, (7,7),0)
    P_YELLOW_MIN = np.array([17, 50, 50],np.uint8)
    P_YELLOW_MAX = np.array([35,255,255],np.uint8)
    mask = cv2.inRange(hsv_img, P_YELLOW_MIN, P_YELLOW_MAX)
    yellow_tresh = cv2.bitwise_and(img,img,mask=mask)
    yellow_tresh = cv2.cvtColor(yellow_tresh,cv2.COLOR_BGR2GRAY)
    yellow_tresh = cv2.bitwise_not(yellow_tresh)

    return yellow_tresh

def findCenter(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    center = x+(w/2)
    return center

def findLCont(img,roi):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    biggest_cnts =sorted(contours, key=cv2.contourArea, reverse=True)
    possible_cnts = []
    for cnt in biggest_cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if ((w<200) and (h/w > 1.25)):
            possible_cnts.append(cnt)

    sorted_conts = possible_cnts[:6]
    sorted_conts = sorted(sorted_conts, key = findCenter)
    
    for cnt in sorted_conts:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("contours", roi)

    
    return sorted_conts
    
def thirdTresh(img,itr):
    
    if (itr ==1):
        img = cv2.GaussianBlur(img, (7,7),0)
        H = 255
        L = 55
    
    if (itr ==2):
        img = cv2.GaussianBlur(img, (13,13),0)
        H = 255
        L = 147        
    ret,thresh1 = cv2.threshold(img,L,H,cv2.THRESH_BINARY)
    return thresh1

def imgShower(img_name,img):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def charRec(char_cnts,img):
    characters = "" 
    height, width = img.shape
    #imgShower("char",img)
    color_img = np.zeros((height, width, 3), np.uint8)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for char in char_cnts:                                         # for each char in plate
        x,y,w,h = cv2.boundingRect(char)

        cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,255,0),2)
        char_roi = img[y:y+h,x:x+w]
        
        res_roi = cv2.resize(char_roi, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        npaROIResized = res_roi.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        characters = characters + strCurrentChar
    #print("plate recognition done")
    return characters

        
def loadKNN():

    npaClassifications = np.loadtxt("classifications.txt", np.float32) 
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(2)                                                             # set default K to 1
    
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object
    print("KNN training done")
    
def orderPoints(hull):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = np.float32)
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = np.zeros((hull.shape[0]), dtype = np.float32)    #make np as big as all the points 
    #print(s.shape)
    for i in range(hull.shape[0]):                      #look in x/y coordinates

        #print(hull[i])
        opt =(np.sum(hull[i]))                          #sum x+y
        s[i] = opt                                      #put sum result in list
    #idx = np.argmax(s)
    rect[0] = hull[np.argmin(s)]
    rect[2] = hull[np.argmax(s)]
    
    t = np.zeros((hull.shape[0]), dtype = np.float32)    #make np as big as all the points 
    for i in range(hull.shape[0]):
        difference = (np.diff(hull[i]))
        t[i] = difference

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference

    rect[1] = hull[np.argmin(t)]
    rect[3] = hull[np.argmax(t)]

	# return the ordered coordinates
    return rect

def fourPointTransform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    	
    (tl, tr, br, bl) = rect
    
     
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
     
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    	[0, 0],
    	[maxWidth - 1, 0],
    	[maxWidth - 1, maxHeight - 1],
    	[0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped
            
            
        
"""                 MAIN            """
loadKNN()
for img in glob.glob("Serie2/*.jpg"):
    img = cv2.imread(img)
    res_img = cv2.resize(img,None, fx = .5, fy = .5)
    shw_img = cv2.resize(res_img,None, fx = .5, fy = .5)
    imgShower("Original Image", shw_img)
    prep_img = prepImg(res_img,1)                                 #thresholding image
    contours,theIndex = biggestCont(prep_img)                   #extraxt biggest contour
    
    hull = hullCont(contours[theIndex])
    #rect,approxCont = makeContPretty(contours[theIndex])        #make a  bounding rectangle
    rect,approxCont = makeContPretty(hull)
    roi = deskewAndCrop(res_img,rect,approxCont)                #makes previous rectangle flat
    imgShower("ROI", roi)
    
#    showStuff(approxCont, res_img, prep_img, roi)               #shows everything
#
#    cv2.imshow("corners" , roi)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
################################################################################################################
    roi_thresh = prepImg(roi,2)
    roi_thresh = thirdTresh(roi_thresh,1)
    contours, theIndex = biggestCont(roi_thresh)
    hull = hullCont(contours[theIndex])
    rect,approxCont = makeContPretty(hull)
    cv2.drawContours(roi,[hull],-1, (0,255,0), 3)
    #imgShower("hopefully", roi)
    rect = orderPoints(hull)
    warped = fourPointTransform(roi,rect)
    imgShower("Warped Image", warped)
################################################################################################################
    warped_copy = warped.copy()
    #roi_res = cv2.resize(roi(100, 50)) 
    #contours = getCharCont(roi_copy)
    warped_tresh = secondTresh(warped)
    warped_tresh = thirdTresh(warped_tresh,2)
    #imgShower("tresh", warped_tresh)
    warped_tc = warped_tresh.copy()
    warped_cnts = findLCont(warped_tresh,warped_copy)
    plate = charRec(warped_cnts, warped_tc)
    print(plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    