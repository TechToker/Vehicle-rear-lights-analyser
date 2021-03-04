from matplotlib import pyplot as plt
import numpy as np
import cv2


def PlotImageRow(list_of_images, titles=None, disable_ticks=False, size=10):
    count = len(list_of_images)
    plt.figure(figsize=(size, size))

    for idx in range(count):
        subplot = plt.subplot(1, count, idx + 1)
        if titles is not None:
            subplot.set_title(titles[idx])

        img = list_of_images[idx]

        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)

        if disable_ticks:
            plt.xticks([]), plt.yticks([])

    plt.show()

car_img = cv2.imread('./testing_data/car_go_2.png', cv2.IMREAD_COLOR)
car_img_rgb = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2YCrCb)

# plate_points = np.array([[656, 565], [1206, 565], [1206, 1047], [656, 1047]], np.int32)
#
# startX = min(plate_points[:, 0])
# finishX = max(plate_points[:, 0])
#
# startY = min(plate_points[:, 1])
# finishY = max(plate_points[:, 1])

#plate_img = car_img[startY: finishY, startX: finishX]

plate_img = car_img.copy()

threshold_img = np.zeros((len(plate_img), len(plate_img[0]))).astype(np.uint8)

for i in range(len(threshold_img)):
    for j in range(len(threshold_img[0])):
        #print(threshold_img[i][j])
        if plate_img[i][j][0] < 100 or plate_img[i][j][1] < 160:
            threshold_img[i][j] = 0
        else:
            threshold_img[i][j] = 255


kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(threshold_img, kernel, iterations=1)

kernel = np.ones((8, 8), np.uint8)
dilation = cv2.dilate(erosion, kernel, iterations = 2)

PlotImageRow([plate_img])
# threshold_img, erosion, dilation
PlotImageRow([threshold_img])
PlotImageRow([erosion])
PlotImageRow([dilation])

print(dilation.shape)

connectivity = 4
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation, connectivity, cv2.CV_32S)

bboxes = []

import sys
np.set_printoptions(threshold=sys.maxsize)

for (i, label) in enumerate(range(1, n_labels)):
    # print()
    # print(labels)
    #

    ymax, xmax = np.max(np.where(labels == label), 1)
    ymin, xmin = np.min(np.where(labels == label), 1)

    print(xmin, ymin, xmax, ymax)

    bboxes.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    #cv2.rectangle(car_img_rgb, (xmin - 10 + plate_points[0][0], ymin - 10 + plate_points[0][1]), (xmax + 10 + plate_points[0][0], ymax + 10 + plate_points[0][1]), (255, 0, 0), 2)
    cv2.rectangle(car_img_rgb, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 0, 0), 2)

    # centroid coordinates
    #cent_x, cent_y = int(centroids[label, 0]), int(centroids[label, 1])
    #cv2.circle(plate_img, (cent_x, cent_y), 3, (255, 0, 0), -1)
    #cv2.putText(plate_img, str(i), (cent_x, cent_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

PlotImageRow([car_img_rgb])