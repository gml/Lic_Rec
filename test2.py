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

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

def prepImg(img):
    #blurred = cv2.GaussianBlur(img, (1,1),0)
    kernel = np.ones((3,3),np.uint8)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #yellow_tresh = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
    mask = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
    yellow_tresh = cv2.bitwise_and(img,img,mask=mask)
    yellow_tresh = cv2.erode(yellow_tresh,kernel,iterations = 10)
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
    W = rect[1][0]#+50
    H = rect[1][1]#+50
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
    size = (x2-x1, y2-y1)#-50/50
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

#def secondThresh(img):
#    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    BLACK_MIN = 60
#    BLACK_MAX = 120
#    mask = cv2.threshold(gray_img,0,60,cv2.THRESH_BINARY)
#    showTest(mask)

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
#    peri = cv2.arcLength(cnt, True)
#    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
#    return approx
    for c in cnt:
    	# approximate the contour
    	peri = cv2.arcLength(c, True)
    	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
     
    	# if our approximated contour has four points, then
    	# we can assume that we have found our screen
    	if len(approx) == 4:
    		screenCnt = approx
    		break   
    return screenCnt
    
def getEdges(img):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.threshold(img,,255,cv2.THRESH_TOZERO)
    #gray_img = img[:,:,1]
    #gray_img = cv2.bilateralFilter(img, 1, 17, 30)
    #kernel = np.ones((15,28),np.uint8)
    #gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN,kernel)
    #kernel = np.ones((27,27),np.uint8)
    #gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT,kernel)
    #L = 45
    #H = 65
    cv2.namedWindow('image')
    cv2.createTrackbar('L','image',0,255,nothing)
    cv2.createTrackbar('H','image',0,255,nothing)
    while(1):
        
        L = cv2.getTrackbarPos('L', 'image')
        H = cv2.getTrackbarPos('H', 'image')
        edged_img = cv2.Canny(img,L,H)
        cv2.imshow("gray image",img)
        cv2.imshow("edged image",edged_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:   # hit escape to quit
            break
    cv2.destroyAllWindows()
    
    edged_img = cv2.Canny(img,L,H)
#    #showing stuff
#    cv2.imshow("gray image",gray_img)
#    cv2.imshow("edged image",edged_img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return edged_img

####################################################################################
def secondTresh(img):
    
    
    
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv_img = cv2.GaussianBlur(hsv_img, (7,7),0)
#    cv2.namedWindow('image')
#    cv2.createTrackbar('L','image',0,255,nothing)
#    cv2.createTrackbar('H','image',0,255,nothing)
#    while(1):    
#        L = cv2.getTrackbarPos('L', 'image')
#        H = cv2.getTrackbarPos('H', 'image')
#        P_YELLOW_MIN = np.array([L, 50, 50],np.uint8)
#        P_YELLOW_MAX = np.array([H,255,255],np.uint8)
#        mask = cv2.inRange(hsv_img, P_YELLOW_MIN, P_YELLOW_MAX)
#        yellow_tresh = cv2.bitwise_and(img,img,mask=mask)
#        yellow_tresh = cv2.cvtColor(yellow_tresh,cv2.COLOR_BGR2GRAY)
#        cv2.imshow("roi_tresh", yellow_tresh)
#        k = cv2.waitKey(1) & 0xFF
#        if k == 27:   # hit escape to quit
#            break
#    cv2.destroyAllWindows()
    P_YELLOW_MIN = np.array([17, 50, 50],np.uint8)
    P_YELLOW_MAX = np.array([35,255,255],np.uint8)
    mask = cv2.inRange(hsv_img, P_YELLOW_MIN, P_YELLOW_MAX)
    yellow_tresh = cv2.bitwise_and(img,img,mask=mask)
    yellow_tresh = cv2.cvtColor(yellow_tresh,cv2.COLOR_BGR2GRAY)
    yellow_tresh = cv2.bitwise_not(yellow_tresh)

    #yellow_tresh = cv2.erode(yellow_tresh,kernel,iterations = 1)
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
        if (w<200):
            possible_cnts.append(cnt)

    sorted_conts = possible_cnts[:6]
    sorted_conts = sorted(sorted_conts, key = findCenter)

        
        
    #print(len(biggest_conts))
            
    
    for cnt in sorted_conts:
        x,y,w,h = cv2.boundingRect(cnt)
#        print("cnt dimensions: w = ", w, "h = ", h)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.drawContours(roi, [cnt], 0, (0,255,255), 3)
        cv2.imshow("contours", roi)

    
    return sorted_conts
    
def thirdTresh(img):
    
    
    img = cv2.GaussianBlur(img, (13,13),0)
    H = 255
    L = 147
    ret,thresh1 = cv2.threshold(img,L,H,cv2.THRESH_BINARY)
    
#    cv2.namedWindow('image')
#    cv2.createTrackbar('L','image',0,255,nothing)
#    cv2.createTrackbar('H','image',0,255,nothing)
#    while(1):
#        
#        L = cv2.getTrackbarPos('L', 'image')
#        H = cv2.getTrackbarPos('H', 'image')
#        ret,thresh1 = cv2.threshold(img,L,H,cv2.THRESH_BINARY)
#        cv2.imshow("gray image",img)
#        cv2.imshow("edged image",thresh1)
#        k = cv2.waitKey(1) & 0xFF
#        if k == 27:   # hit escape to quit
#            break
#    cv2.destroyAllWindows()

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
        #imgShower("res_roi", res_roi)
        
        #print(res_roi.shape)
        
        npaROIResized = res_roi.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        characters = characters + strCurrentChar
    print("plate recognition done")
    return characters

        
def loadKNN():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    npaClassifications = np.loadtxt("classifications.txt", np.float32) 
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(1)                                                             # set default K to 1
    
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # train KNN object
    print("KNN training done")
    
        
"""                 MAIN            """
loadKNN()
for img in glob.glob("Serie2/*.jpg"):
    img = cv2.imread(img)
    res_img = cv2.resize(img,None, fx = .5, fy = .5)
    prep_img = prepImg(res_img)                                 #thresholding image
    contours,theIndex = biggestCont(prep_img)                   #extraxt biggest contour
    
    hull = hullCont(contours[theIndex])
    #rect,approxCont = makeContPretty(contours[theIndex])        #make a  bounding rectangle
    rect,approxCont = makeContPretty(hull)
    roi = deskewAndCrop(res_img,rect,approxCont)                #makes previous rectangle flat
    if (SHOW_STEPS == 1):
        showStuff(approxCont, res_img, prep_img, roi)               #shows everything
    
        cv2.imshow("corners" , roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
################################################################################################################
    roi_copy = roi.copy()
    #roi_res = cv2.resize(roi(100, 50)) 
    #contours = getCharCont(roi_copy)
    roi_tresh = secondTresh(roi)
    roi_tresh = thirdTresh(roi_tresh)
    imgShower("tresh", roi_tresh)
    roi_tc = roi_tresh.copy()
    roi_cnts = findLCont(roi_tresh,roi_copy)
    plate = charRec(roi_cnts, roi_tc)
    print(plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
#    kernel = np.ones((3,3),np.uint8) 
#    roi_closed = cv2.morphologyEx(roi_tresh, cv2.MORPH_CLOSE, kernel,iterations = 1)
#    imgShower("closed",roi_closed)
#    roi_edged = getEdges(roi_closed)
    
#    contours = findLCont(roi_tresh)
#    im2, contours2, hierarchy = cv2.findContours(roi_closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
#    print (len(contours2))
#    cv2.drawContours(roi_closed, im2, -1, (0,255,0), 3)
#    cv2.imshow("roi_closed", roi_closed)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#part 2 begins here 
"""   
    prep_roi = prepImg(roi)                     #yellow treshold
    cv2.imshow("prep", prep_roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    edged_roi = getEdges(prep_roi)              #get edges
    roi_cnts, roi_idx = biggestCont(edged_roi)  #get biggest cont
    roi_hull = hullCont(roi_cnts[roi_idx])
    roi_rect,roi_cnt = makeContPretty(roi_hull)
    print(roi_cnt)
    #cv2.drawContours(roi, roi_cnts[roi_idx], -1, (0,0,255), 3)
    cv2.drawContours(roi, (roi_cnts[roi_idx]), -1, (0,0,255), 3)
    cv2.imshow("Game Boy Screen", roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(roi_cnts)
#    fr_pnt_cont = frptConts(roi_cnts)
#    
#    showTest(roi,fr_pnt_cont)
#    print("4pt cont: ",fr_pnt_cont)
#    roi_hull = hullCont(roi_cnts[roi_idx])
##    roi_rect, roi_hull_cont = makeContPretty(roi_hull)
#    final_roi = deskewAndCrop(roi,roi_rect,roi_hull_cont)
#    showTest(final_roi,roi_hull_cont)
"""   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    