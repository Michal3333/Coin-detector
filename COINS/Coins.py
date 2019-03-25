import colorsys

import imutils as imutils
from skimage import data, io, filters, exposure, img_as_ubyte, img_as_float
from skimage.color import rgb2gray, hsv2rgb, rgb2hsv
import skimage.morphology as mp
from skimage import measure
import numpy as np
from skimage.feature import canny
from skimage.filters.edges import convolve
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
import cv2




def config1(img2):
    # one config which led me to a nice solution
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY);
    img2 = cv2.medianBlur(img2, 7)
    img2 = cv2.multiply(img2, np.array([0.8]))
    img2 = cv2.GaussianBlur(img2, (7, 5), 0)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mean = np.mean(img2)
    std = np.std(img2)
    img2 = cv2.Canny(img2, 0, mean + std)
    img2 = cv2.dilate(img2, el, iterations=1)
    return img2

def config2(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_gray = cv2.GaussianBlur(img_gray, (7, 5), 0)
    img_gray = cv2.fastNlMeansDenoising(img_gray, 10, 10, 7, 21)
    img_canny = canny(img_gray)
    img_canny = (img_canny > 0.04) * 1
    img_sobel = filters.sobel(img_gray)
    img_sobel = (img_sobel > 0.04) * 1
    sum = np.logical_or(img_sobel, img_canny)
    sum = np.array(sum, dtype=np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    sum = cv2.dilate(sum, el, iterations=2)
    return sum
def colorPoints(center, radius):
    points = []
    radius = radius *0.92
    # print(center[0])
    points.append([center[0], center[1]])

    points.append([center[0] + int(0.1 * radius), center[1]])
    points.append([center[0] - int(0.1 * radius), center[1]])
    points.append([center[0], center[1] + int(0.1 * radius)])
    points.append([center[0], center[1] - int(0.1 * radius)])

    points.append([center[0] + int(0.8 * radius), center[1]])
    points.append([center[0] - int(0.8 * radius), center[1]])
    points.append([center[0], center[1] + int(0.8 * radius)])
    points.append([center[0], center[1] - int(0.8 * radius)])

    return points

def getArea(img , point):
    newImg = img[(point[1]-5):(point[1]+5), (point[0]-5):(point[0]+5) ]
    return  newImg
def isGray(colors):
    # print(str(colors[0]) + " " + str(colors[1]) +" "+ str(colors[2]))
    # print(s)
    # print(hsv)
    if colors[1] <= 0.27 :
        return True
    else:
        return False

def isGold(colors):
    if (colors[0] > 0.036) and (colors[0] < 0.167) and (0.8 > colors[1] > 0.24):
        return True
    else:
        return False
def is5(colors):
    test = 0
    for i in range(4):
        if colors[0][i+1][1] - colors[1][i][1] > 0.10:
            test+=1

    if test >=3 :
        return True
    else:
        return False
def is2(colors):
    test = 0
    for i in range(4):
        if colors[1][i][1] -  colors[0][i+1][1] > 0.1 :
            test+=1

    if test >=3 :
        return True
    else:
        return False

def blackwhite(img):
    retuenImg = np.zeros_like(img)
    img5, contours, hierarhy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = img
    for k, x in enumerate(contours):
        cv2.drawContours(img2, [x], 0, np.asarray(colorsys.hsv_to_rgb(k / len(contours), 1, 1)) * 255, 7)


    for k, contour in enumerate(contours):
        if(cv2.contourArea(contour)>40):


            # print(contour[0,0,3])

            temp = np.zeros_like(img)
            # print(temp[0][0])
            cv2.drawContours(temp, [contour], 0, np.asarray(colorsys.hsv_to_rgb(k/len(contours),1,1))*255, 7)

            # temp[contour[:,0,1],contour[:,0,0]] = 255
            # temp = ndi.binary_fill_holes(temp)
            temp = cv2.fillPoly(temp,[contour],(255,255,255))
            retuenImg = np.logical_or(retuenImg, temp)
    # retuenImg = ndi.binary_fill_holes(retuenImg)
    return retuenImg

def rozdzielanie(img):
    img = rgb2gray(img)
    distance = ndi.distance_transform_edt(img)
    plt.imsave('results/dis.jpg', distance, cmap=plt.cm.gray)
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

def rozpoznawanieKoloru(info, img):
    points = colorPoints(info[0],info[1])
    srodek = []
    zew = []
    for nr,i in enumerate(points):
        newimg = getArea(img, i)
        color = [np.mean(newimg[:, :, 0]), np.mean(newimg[:, :, 1]), np.mean(newimg[:, :, 2])]
        if nr <= 4 :
            srodek.append(colorsys.rgb_to_hsv(color[0],color[1],color[2]))
        else:
            zew.append(colorsys.rgb_to_hsv(color[0],color[1],color[2]))
        # img = cv2.circle(img,(i[0],i[1]) , 20, (255, 0, 0), 2)

    newimg = getArea(img, points[0])

    return [srodek, zew]


def momenty(img, contours):
    info = []
    count = 0
    xd = np.zeros_like(img)
    # print(img[0,0])
    for nr, contour in enumerate(contours):
        moments = cv2.HuMoments(cv2.moments(contour))
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)

        if  (len(approx) > 8) and (abs(w-h) < 20) and (area > 400) and (abs(radius * radius * np.pi - area) < radius * radius * np.pi * 0.2 ):
            count+=1
            # img = cv2.circle(img, center, radius,(255, 0, 0),2)
            info.append([center,radius,contour])
            # cv2.drawContours(xd, [contour], 0, np.asarray(colorsys.hsv_to_rgb(nr / len(contours), 1, 1)) * 255, 7)

    # print(count)
    return info

def rozpoznawnieMonet(info, img):
    suma =0
    dwa = False
    dwaRadius = 0
    piec = False
    piecRadius = 0
    piecG =False
    piecGRadius = 0
    srebrne = []
    for object in info:
        colors1 = rozpoznawanieKoloru(object, img)
        if isGold(colors1[0][0]):
            if is5(colors1):
                img = cv2.circle(img, object[0], int(0.92 * object[1]), (0, 0, 255), 2)
                piec = True
                piecRadius = int(0.92 * object[1])
                suma+=5
                print(suma)
            else:
                img = cv2.circle(img, object[0], int(0.92 * object[1]), (0, 255, 0), 2)
                piecG = True
                piecGRadius = int(0.92 * object[1])
                suma += 0.05
                print(suma)
        elif isGray(colors1[0][0]):
            if is2(colors1):
                img = cv2.circle(img, object[0], int(0.92 *object[1]), (255, 255, 0), 2)
                dwa = True
                dwaRadius = int(0.92 * object[1])
                suma+= 2
                print(suma)
            else:
                srebrne.append(object)
                # img = cv2.circle(img, object[0], int(0.92 * object[1]), (255, 0, 0), 2)
    nierozpoznane = False
    if len(srebrne) > 0:
        maxRadius = srebrne[0][1]
        minRadius = srebrne[0][1]
        for i in srebrne:
            if i[1] > maxRadius:
                maxRadius = i[1]
            if i[1] < maxRadius:
                minRadius = i[1]
        if minRadius/maxRadius > 0.9:
            jednasrebrna = True
        else:
            jednasrebrna = False
        print(str(jednasrebrna) + " " + str(minRadius) + " " + str(maxRadius))
        if jednasrebrna:
            if piec:
                if maxRadius / piecRadius > 0.9:
                    for object in srebrne:
                        img = cv2.circle(img, object[0], int(0.92 * object[1]), (255, 0, 255), 2)
                        suma+=1
                        print(suma)
                else:
                    for object in srebrne:
                        img = cv2.circle(img, object[0], int(0.92 * object[1]), (0, 255, 255), 2)
                        suma+=0.1
                        print(suma)
            elif dwa:
                if dwaRadius/maxRadius > 0.9:
                    for object in srebrne:
                        img = cv2.circle(img, object[0], int(0.92 * object[1]), (255, 0, 255), 2)
                        suma+=1
                        print(suma)
                else:
                    for object in srebrne:
                        img = cv2.circle(img, object[0], int(0.92 * object[1]), (0, 255, 255), 2)
                        suma+=0.1
                        print(suma)
            elif piecG:
                if piecGRadius/maxRadius > 0.9:
                    for object in srebrne:
                        img = cv2.circle(img, object[0], int(0.92 * object[1]), (0, 255, 255), 2)
                        suma+=0.1
                        print(suma)
                else:
                    for object in srebrne:
                        img = cv2.circle(img, object[0], int(0.92 * object[1]), (255, 0, 255), 2)
                        suma+=1
                        print(suma)
            else:
                for object in srebrne:
                    img = cv2.circle(img, object[0], int(0.92 * object[1]), (255, 0, 0), 2)

                    nierozpoznane = True

        else:
            for object in srebrne:
                if object[1]/maxRadius > 0.9:
                    img = cv2.circle(img, object[0], int(0.92 * object[1]), (255, 0, 255), 2)
                    suma+=1
                    print(suma)
                else:
                    img = cv2.circle(img, object[0], int(0.92 * object[1]), (0, 255, 255), 2)
                    suma+= 0.1
                    print(suma)
    if nierozpoznane:
        print("od " + str(suma + len(srebrne)*0.1) +" do "+ str(suma + len(srebrne)*1))
    else:
        print("suma to:")
        print(suma)



    plt.imsave('results/wynik.jpg', img, cmap=plt.cm.gray)



img = data.imread('photos/SAM_1681.jpg')
img = img[0]



    # img = img ** 2

    # print(img)
pyr = cv2.pyrMeanShiftFiltering(img, 7, 15)
pyr = img

img_gray = cv2.cvtColor(pyr,cv2.COLOR_RGB2GRAY)
# img_gray = cv2.GaussianBlur(img_gray, (7, 5), 0)
img_gray = cv2.fastNlMeansDenoising(img_gray, 10, 10, 7, 21)
img_canny = canny(img_gray,2)

img_canny = (img_canny > 0.04) * 1

img_sobel = filters.sobel(img_gray)

img_sobel = (img_sobel > 0.04) * 1
sum = np.logical_or(img_sobel, img_canny)
sum = np.array(sum, dtype=np.uint8)
el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
sum = cv2.dilate(sum, el, iterations=2)

# sum = ndi.binary_fill_holes(sum)



imgzjecie = img

# sum = config1(img)
print("init")
sum = blackwhite(sum)
# print("black")
plt.imsave('results/init2.jpg', sum, cmap=plt.cm.gray)

#sprawko

# sum = rgb2gray(sum)
# contours = measure.find_contours(sum, 0.8)
#
# fig, ax = plt.subplots()
# for n, contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
# ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
# plt.savefig('1samolot.jpg')




contours = rozdzielanie(sum)
# print(len(contours))
imgtemp =img
# print(countures[0])
# for i, contour in enumerate(contours):
#     cv2.drawContours(imgtemp, [contour], 0, np.asarray(colorsys.hsv_to_rgb(i / len(contours), 1, 1)) * 255, 7)


info = momenty(pyr, contours)
rozpoznawnieMonet(info, pyr)

# plt.imsave('5test.jpg', pyr, cmap=plt.cm.gray)
# info = momenty(pyr, contours)
# for i in info:
#     rozpoznawanieKoloru(i,pyr)





# (len(approx) > 8) and (abs(w-h) < 20) and (area > 200) and (abs(radius * radius * np.pi - area) < radius * radius * np.pi *0.2 )
