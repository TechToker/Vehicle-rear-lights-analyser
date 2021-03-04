import numpy as np
import cv2
import sys


def TailDetector(img):
    car_img_rgb = img.copy()
    car_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    #plate_img = car_img.copy()
    threshold_img = np.zeros((len(car_img), len(car_img[0]))).astype(np.uint8)

    for i in range(len(threshold_img)):
        for j in range(len(threshold_img[0])):
            if car_img[i][j][0] < 80 or car_img[i][j][1] < 160:
                threshold_img[i][j] = 0
            else:
                threshold_img[i][j] = 255

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(threshold_img, kernel, iterations=1)

    kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    connectivity = 4
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation, connectivity, cv2.CV_32S)

    bboxes = []
    np.set_printoptions(threshold=sys.maxsize)

    for (i, label) in enumerate(range(1, n_labels)):
        ymax, xmax = np.max(np.where(labels == label), 1)
        ymin, xmin = np.min(np.where(labels == label), 1)

        #print(xmin, ymin, xmax, ymax)

        bboxes.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        cv2.rectangle(car_img_rgb, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 0, 0), 2)

    return car_img_rgb, np.array(bboxes)