import cv2
import numpy as np
import pytesseract
import os
import  MultiDigits

per = 25
roi =  [[(402, 82), (636, 118), ' text', 'answ1'], [(400, 126), (638, 164), ' text', 'answ2']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

startImage = cv2.imread('Images/BlankTemplate3.png')
h,w,c = startImage.shape
##detector creating
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(startImage,None)#unique elements of image, the representation of kp(easier to comp to understand and differenciate between them)
#imKp1 = cv2.drawKeypoints(startImage, kp1, None)
#cv2.imshow('KeyPoints', imKp1)

path = 'Images/BlankTesting'

myPicLIst = os.listdir(path)
for j, y in enumerate(myPicLIst):
    img = cv2.imread(path + '/'+ y)

    #cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1) # matching the decreptors
    matches.sort(key= lambda x :x.distance)#sort all the matches based on a distance (the lower the distance, the better the match is)
    good = matches [:int(len(matches)*(per/100))]#extract 25% of the best matches
    imgMatch = cv2.drawMatches(img,kp2,startImage,kp1,good[:20],None, flags =2)
    cv2.imshow(y, imgMatch)

    # finding the relationships between start image and a test image

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))  # allign form
    #cv2.imshow(y, imgScan)
    imgShow = imgScan.copy() #final image
    imgMask = np.zeros_like(imgShow)

    myData = [] # data for each of our images

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), (r[1][1])), (0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask,0.1, 0)

        #Cropping
        imgCrop = imgScan[r[0][1]: r[1][1], r[0][0]:r[1][0]]
        answer = MultiDigits.image_processing(imgCrop)
        cv2.imshow(str(x) , imgCrop)
        #print(f'{r[3]}:{pytesseract.image_to_string(imgCrop)}')
        print(answer)
        print('------------')

        #myData.append(pytesseract.image_to_string(imgCrop))


    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    cv2.imshow(y + '2', imgShow)

#cv2.imshow('Output', startImage)
cv2.waitKey(0)