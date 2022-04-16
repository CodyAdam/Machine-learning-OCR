def HomographyTransform(originalImg, targetImg, rectPts):

    sourcePts = np.array(rectPts)
    h, w = originalImg.shape
    targetPts = np.array([[0, 0], [0, h], [w, h], [w, 0]])

    h, status = cv2.findHomography(sourcePts, targetPts)

    imgOutput = cv2.warpPerspective(
        targetImg, h, (originalImg.shape[1], originalImg.shape[0]))

    return imgOutput


def GetHomographyRect(original, target, minimunMatch):
    originalImg = LoadImgGrey(original)
    targetImg = LoadImgGrey(target)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(originalImg, None)
    keypoints2, descriptors2 = sift.detectAndCompute(targetImg, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matches = cv2.FlannBasedMatcher(index_params,
                                    search_params).knnMatch(descriptors1,
                                                            descriptors2,
                                                            k=2)

    goodMatch = []
    for x, y in matches:
        if x.distance < 0.8 * y.distance:
            goodMatch.append(x)

    if len(goodMatch) > minimunMatch:
        originalPts = np.float32(
            [keypoints1[x.queryIdx].pt for x in goodMatch]).reshape(-1, 1, 2)
        targetPts = np.float32([keypoints2[x.trainIdx].pt
                                for x in goodMatch]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(originalPts, targetPts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = originalImg.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        output = HomographyTransform(originalImg, LoadImg(target), dst)
        targetImg = cv2.polylines(targetImg, [np.int32(dst)], True, 255, 2,
                                  cv2.LINE_AA)

    else:
        matchesMask = None
        output = targetImg

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)

    imgMatches = cv2.drawMatches(originalImg, keypoints1, targetImg,
                                 keypoints2, goodMatch, None, **draw_params)

    cv2.imwrite("HomographyMatches.png", imgMatches)
    cv2.imwrite("WrapedImage.png", output)
    return output
