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
            conture = measure.find_contours(temp, 0.8)
            contures.append(conture[0])
    return contures