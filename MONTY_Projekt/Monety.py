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

temp1 = data.imread('zdjecia/1.jpg')
images = [temp1]
for im_number, img in enumerate(images):
    img_gray = rgb2gray(img)

    plt.imsave(str(im_number) + '_grey.jpg',  img_gray , cmap=plt.cm.gray)


