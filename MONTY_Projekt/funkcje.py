def rozdzielanie(img):
    img = rgb2gray(img)
    distance = ndi.distance_transform_edt(img)
    localMax = peak_local_max(distance, indices=False, min_distance=20, labels=img)
    markers = ndi.label(localMax, structure=np.ones((3, 3)))[0]
    labels = mp.watershed(-distance, markers, mask=img)
    contures = []
    for l in np.unique(labels):
        if l != 0:
            temp = np.zeros_like(img)
            temp[l == labels] = 255
            temp = np.array(temp, dtype=np.uint8)
            im5, conture, h = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contures.append(conture[0])
    return contures

def blackwhite(img):
    retuenImg = np.zeros_like(img)
    img5, contours, hierarhy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = img
    for k, x in enumerate(contours):
        cv2.drawContours(img2, [x], 0, np.asarray(colorsys.hsv_to_rgb(k / len(contours), 1, 1)) * 255, 7)
    for k, contour in enumerate(contours):
        if(cv2.contourArea(contour)>40):
            temp = np.zeros_like(img)
            cv2.drawContours(temp, [contour], 0, np.asarray(colorsys.hsv_to_rgb(k/len(contours),1,1))*255, 7)
            temp = ndi.binary_fill_holes(temp)
            retuenImg = np.logical_or(retuenImg, temp)
    retuenImg = ndi.binary_fill_holes(retuenImg)
    return retuenImg

def momenty(img, contours):
    info = []
    xd = np.zeros_like(img)
    for nr, contour in enumerate(contours):
        moments = cv2.HuMoments(cv2.moments(contour))
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if (len(approx) > 8) and (abs(w-h) < 20) and (area > 200) and (abs(radius * radius * np.pi - area) < radius * radius * np.pi *0.2 ) :
            img = cv2.circle(img, center, radius,(255, 0, 0),2)
            info.append([center,radius,contour])
            cv2.drawContours(xd, [contour], 0, np.asarray(colorsys.hsv_to_rgb(nr / len(contours), 1, 1)) * 255, 7)
    return info
