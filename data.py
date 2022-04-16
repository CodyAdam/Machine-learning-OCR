#   dependencies :
#       python 3.7.4
#       opencv-python 3.4.2.16
#       opencv-contrib-python 3.4.2.16

import random
import os
import numpy as np
import cv2


def LoadImg(path):
    img = cv2.imread(path)
    if type(img) != np.ndarray:
        print("\nImage not found! - "+ path +"\n")
        exit()
    return img


def LoadImgGrey(path):
    img = cv2.imread(path, 0)
    if type(img) != np.ndarray:
        print("\nImage not found! - "+ path +"\n")
        exit()
    return img


def HomographyTransform(originalImg, targetImg, rectPts):

    sourcePts = np.array(rectPts)
    h, w = originalImg.shape
    targetPts = np.array([[0, 0], [0, h], [w, h], [w, 0]])

    h, status = cv2.findHomography(sourcePts, targetPts)

    imgOutput = cv2.warpPerspective(
        targetImg, h, (originalImg.shape[1], originalImg.shape[0]))

    return imgOutput


def GetHomography(originalPath, targetPath, minimunMatch):
    originalImg = LoadImgGrey(originalPath)
    targetImg = LoadImgGrey(targetPath)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(originalImg, None)
    keypoints2, descriptors2 = sift.detectAndCompute(targetImg, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(descriptors1, descriptors2, k=2)

    goodMatch = []
    for x, y in matches:
        if x.distance < 0.8 * y.distance:
            goodMatch.append(x)

    if len(goodMatch) > minimunMatch:
        originalPts = np.float32(
            [keypoints1[x.queryIdx].pt for x in goodMatch]).reshape(-1, 1, 2)
        targetPts = np.float32([keypoints2[x.trainIdx].pt for x in goodMatch]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(originalPts, targetPts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = originalImg.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        output = HomographyTransform(originalImg, LoadImg(targetPath), dst)
        targetImg = cv2.polylines(targetImg, [np.int32(dst)], True, 255, 2, cv2.LINE_AA)
        succeed = True

    else:
        matchesMask = None
        output = LoadImg(targetPath)
        succeed = False

    draw_params = dict(matchColor=(100, 255, 50), singlePointColor=None, matchesMask=matchesMask, flags=2)

    imgMatches = cv2.drawMatches(originalImg, keypoints1, targetImg, keypoints2, goodMatch, None, **draw_params)
    if(succeed):
        cv2.putText(imgMatches, "Homography succeed", (10, imgMatches.shape[0] - 20), 0, 0.7, (20, 255, 0), 2)
    else:
        cv2.putText(imgMatches, "Homography failed", (10, imgMatches.shape[0] - 20), 0, 0.7, (20, 20, 255), 2)
    cv2.imwrite("output/HomographyMatches.png", imgMatches)
    cv2.imwrite("output/WrapedImage.png", output)
    return output, succeed


def CreateMask(img, lower1, upper1, dilate, doShow):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(lower1)
    upper = np.array(upper1)
    mask = cv2.inRange(hsvImg, lower, upper)

    mask = cv2.dilate(mask, (np.ones((dilate, dilate), np.uint8)))
    if doShow:
        cv2.namedWindow("mask", cv2.WINDOW_FREERATIO)
        cv2.imshow("mask", mask)
    cv2.imwrite("output/BinaryMask.png", mask)
    return mask


def Training(roiPath):
    samples = np.empty((0, 100))
    responses = []
    
    for digit in range(0,10):
        for filePath in os.listdir(roiPath+"/"+str(digit)):
            if filePath.endswith(".png"):
                filePath = roiPath+ "/"+ str(digit) + "/" + filePath
                responses.append(digit)
                roi = LoadImgGrey(filePath)
                sample = roi.reshape((1, 100))
                samples = np.append(samples, sample, 0)
                samples = np.float32(samples)
                print(str(digit) + "  -  " + filePath)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size,))
    print("\nTraining complete\n")
    return (samples, responses)


def GetDigitRecogn(samples, responses, sourceImg, mask, doShow):

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contouredImg = sourceImg.copy()
    digit = []
    digitPos = []

    for contour in contours:
        if cv2.contourArea(contour) > 20 and cv2.contourArea(contour) < 1000:
            [x, y, w, h] = cv2.boundingRect(contour)
            if h > 10 and w < h:
                cv2.rectangle(contouredImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
                roi = mask[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                _, results, _, _ = model.findNearest(roismall, k=1)
                digitValue = str(int((results[0][0])))
                digit.append(digitValue)
                digitPos.append([x,y])
                cv2.putText(contouredImg, digitValue, (x, y + h + 13), 0, 0.5, (255, 255, 0), 1)
    if doShow:
        cv2.namedWindow("output", cv2.WINDOW_FREERATIO)
        cv2.imshow("output", contouredImg)
    cv2.imwrite("output/FinalResult.png", contouredImg)
    return digit, digitPos

def SaveRoi(img, mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.namedWindow("roi",cv2.WINDOW_FREERATIO)
    cv2.namedWindow("norm",cv2.WINDOW_FREERATIO)
    keys = [i for i in range(48, 58)]
    for contour in contours:
        if cv2.contourArea(contour) > 25:
            [x, y, w, h] = cv2.boundingRect(contour)

            if h > 10 and h > w:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = mask[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                cv2.imshow("roi", roismall)
                cv2.imshow("norm", img)

                key = cv2.waitKey(0)

                if key == 27:
                    exit()
                elif key in keys:
                    response = int(chr(key))
                    cv2.imwrite("samples/roi/"+str(response)+"/"+str(random.randint(0,10000))+".png", roismall)
    print("\nRoi Printed\n")


def getDistSide(x1,y1,x2,y2):  
     dist = np.sqrt((x2 - x1)**2 + ((3*(y2 - y1))**2))  
     return dist  


def GuessHour(digit, digitPos):
    index = []
    arrangedDigit = []
    refX = 0
    refY = 0
    for _ in range(0, 4):
        next = 0
        for i in range(0, len(digit)):
            if (getDistSide(refX, refY, digitPos[i][0], digitPos[i][1])) < (getDistSide(refX, refY, digitPos[next][0],digitPos[next][1])) and (i not in index)  :
                next = i
        if (index == []):
            refX = digitPos[next][0]
            refY = digitPos[next][1]
        index.append(next)
        dist = getDistSide(0,0,digitPos[next][0],digitPos[next][1])
        arrangedDigit.append(digit[next])

    if len(arrangedDigit) == 4:
        stringDigit = arrangedDigit[0]+arrangedDigit[1]+" h "+arrangedDigit[2] + arrangedDigit[3]
    elif len(arrangedDigit) == 3:
        stringDigit = arrangedDigit[0]+" h "+arrangedDigit[1] + arrangedDigit[2]
    else: stringDigit = "?? h ??"

    result = np.zeros(shape=[200, 500, 3], dtype=np.uint8)
    cv2.putText(result, stringDigit, (20, 127), 0, 3, (255, 255, 255), 2)
    cv2.imwrite("output/HourGuessed.png", result)
    return arrangedDigit


def RecognizeDiggit(targetImg):
    cv2.imwrite("output/Original.png", LoadImg(targetImg))
    img, succeed = GetHomography(HOMOGRAPHY_BASE, targetImg, 60)
    if succeed:
        mask = CreateMask(img, [0, 0, 230], [180, 255, 255], 4, False)
    else:
        mask = CreateMask(img, [0, 0, 254], [180, 255, 255], 4, False)

    digit, digitPos = GetDigitRecogn(samples, responses, img, mask, True)
    print(GuessHour(digit, digitPos))
    

def TestDirectory(pathDir, fileExtension):
    for filePath in os.listdir(pathDir):
        if filePath.endswith(fileExtension):
            filePath = pathDir + "/" + filePath
            RecognizeDiggit(filePath)
            cv2.waitKey(0)


############################################ ACTIONS #################################################

HOMOGRAPHY_BASE = "homography.png"
IMAGE_TO_RECOGNIZE = "samples/example2.jpg"

samples, responses = Training("samples/roi")

# RecognizeDiggit(IMAGE_TO_RECOGNIZE)
TestDirectory("samples", ".jpg")