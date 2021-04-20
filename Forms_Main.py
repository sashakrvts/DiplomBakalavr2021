import cv2
import numpy as np
import pytesseract
import os
import  MultiDigits

def blank_crop(img):
    per = 25
    roi =  [[(402, 78), (634, 120), ' text', 'answer11'],
            [(668, 78), (886, 118), 'text', 'answer12'],
            [(402, 122), (638, 162), 'text', 'answer21'],
            [(670, 126), (888, 164), 'text', 'answer22']]


    answers = []
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    startImage = cv2.imread('Images/BlankTemplate3.png')
    h,w,c = startImage.shape
    ##detector creating
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(startImage,None)#unique elements of image, the representation of kp(easier to comp to understand and differenciate between them)
    #imKp1 = cv2.drawKeypoints(startImage, kp1, None)
    #cv2.imshow('KeyPoints', imKp1)

    #path = 'Images/BlankTesting'

    #myPicLIst = os.listdir(path)
    #for j, y in enumerate(myPicLIst):
    print("blank crop 1", type(img))
    #img = cv2.imread(img)
    print("blank crop 2", type(img))
        #cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1) # matching the decreptors
    matches.sort(key= lambda x :x.distance)#sort all the matches based on a distance (the lower the distance, the better the match is)
    good = matches [:int(len(matches)*(per/100))]#extract 25% of the best matches
    imgMatch = cv2.drawMatches(img,kp2,startImage,kp1,good[:20],None, flags =2)
    #cv2.imshow("match", imgMatch)

        # finding the relationships between start image and a test image

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))  # allign form
    #cv2.imshow("Scan", imgScan)
    imgShow = imgScan.copy() #final image
    imgMask = np.zeros_like(imgShow)

    myData = [] # data for each of our images

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), (r[1][1])), (0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask,0.1, 0)

        #Cropping
        imgCrop = imgScan[r[0][1]: r[1][1], r[0][0]:r[1][0]]
        answer = MultiDigits.image_processing(imgCrop) #string
        #cv2.imshow(str(x) , imgCrop)
        answers.append(answer)

    answers[0]= answers[0]+'.'+answers[1]
    answers[1] = answers[2] + '.' + answers[3]
    answers.remove(answers[3])
    answers.remove(answers[2])

    #cv2.imshow('2', imgShow)
    return answers
    #return answers
        #cv2.imshow(str(x) , imgCrop)
        #print(f'{r[3]}:{pytesseract.image_to_string(imgCrop)}')
        #print(answer)
        #print('------------')

        #myData.append(pytesseract.image_to_string(imgCrop))


    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    #cv2.imshow(y + '2', imgShow)
#img = 'Images/BlankTesting/blank_test1.jpg'
#print(img.type)
img = 'Images/BlankTesting/blank_test2.jpg'
img = cv2.imread(img)
print(blank_crop(img))

#cv2.imshow('Output', startImage)
cv2.waitKey(0)