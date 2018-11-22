def config1(img2):
	#one config which led me to a nice solution
 	img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY);
 	img2 = cv.medianBlur(img2, 7)
 	img2 = cv.multiply(img, np.	array([0.8]))
 	img2 = cv.GaussianBlur(img2, (7, 5), 0)
 	el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
 	img2 = cv.Canny(img2,0,mean+std)
 	img2 = cv.dilate(img2, el, iterations = 1)