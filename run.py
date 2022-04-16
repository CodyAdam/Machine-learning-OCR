#python 3.7.4
#cv2 4.1.1
import numpy as np
import cv2


def LoadImg(path):
    img = cv2.imread(path)
    if type(img) != np.ndarray:
        print("\nImage not found!\n")
        exit()
    return img


def CreateMask(source, lower1, upper1, dilate, doShow):
    lower = np.array(lower1)
    upper = np.array(upper1)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.dilate(mask, (np.ones((dilate, dilate), np.uint8)))
    if doShow:
        cv2.namedWindow("mask", cv2.WINDOW_FREERATIO)
        cv2.imshow("mask", mask)
    return mask


def LoadDataML(samplesPath, responsesPath):
    samples = np.loadtxt(samplesPath, np.float32)
    responses = np.loadtxt(responsesPath, np.float32)
    responses = responses.reshape((responses.size, 1))
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def DigitRecogn(sourceImg, mask, modelML):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    outputImg = np.zeros(sourceImg.shape,
                         np.uint8)  #make a black img from source
    contouredImg = sourceImg.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 10 and w < h:
                cv2.rectangle(contouredImg, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                roi = mask[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                _, results, _, _ = modelML.findNearest(roismall, k=1)
                digit = str(int((results[0][0])))
                cv2.putText(outputImg, digit, (x, y + h), 0, 1, (0, 255, 0), 2)
    return contouredImg, outputImg


####################################################################################

source = LoadImg("training_samples/12.jpg")
hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
mask = CreateMask(hsv, [0, 0, 255], [180, 255, 255], [0, 0, 0], [0, 0, 0], 3,
                  True)
model = LoadDataML("generalsamples.data", "generalresponses.data")
contoured, output = DigitRecogn(source, mask, model)

cv2.imshow("contoured", contoured)
cv2.imshow("output", output)
cv2.waitKey(0)