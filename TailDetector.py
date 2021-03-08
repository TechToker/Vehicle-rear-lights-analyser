from matplotlib import pyplot as plt
import numpy as np
import cv2



def generate_colors(num):
    #r = lambda: np.random.randint(0, 255)
    #return [(r(), r(), r()) for _ in range(num)]

    return [(208, 86, 93), (197, 162, 4), (233, 43, 131), (203, 208, 223), (18, 121, 41), (64, 85, 147),
            (206, 187, 204), (36, 72, 148), (158, 11, 209), (36, 1, 154), (96, 53, 119), (230, 60, 218)]


def SymmetryTest(img, n_labels, labels, stats, centroids):
    source_img = img.copy()
    cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

    labeled_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    light_pairs = []

    colors = generate_colors(n_labels)

    for (i, label) in enumerate(range(1, n_labels)):
        # centroid coordinates
        cent_x, cent_y = int(centroids[label, 0]), int(centroids[label, 1])
        source_img[labels == label] = colors[i][0]

        cv2.circle(source_img, (cent_x, cent_y), 3, (128, 128, 128), -1)
        cv2.putText(source_img, f"{label};{cent_x, cent_y}", (cent_x, cent_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    for i in range(1, len(centroids)):
        for j in range(i + 1, len(centroids)):
            i_cent_x, i_cent_y = int(centroids[i, 0]), int(centroids[i, 1])
            j_cent_x, j_cent_y = int(centroids[j, 0]), int(centroids[j, 1])

            # Distance between left and right lights
            dist_between_lights = 10

            if abs(i_cent_y - j_cent_y) < dist_between_lights:
                #print(f"Find a pair: {i}, {j}; dist:{i_cent_y}; {j_cent_y}; Color: {colors[i]}")

                labeled_image[labels == i, :] = colors[i]
                labeled_image[labels == j, :] = colors[i]

                cv2.putText(labeled_image, f"P: {i, j}", (i_cent_x, i_cent_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                cv2.putText(labeled_image, f"P: {i, j}", (j_cent_x, j_cent_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

                light_pairs.append([i, j])

                # cv2.imshow("Image3", img)
                # cv2.imshow("Image2", labeled_image)
                # return light_pairs

    # cv2.imshow("Image21", labeled_image)
    return light_pairs


def GetThresholdImg(img):
    car_img_Y = img[:, :, 0]
    car_img_Cr = img[:, :, 1]
    car_img_Cb = img[:, :, 2]

    block_size = 15
    c_value = 7
    th_Y = cv2.adaptiveThreshold(car_img_Y, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value)
    th_Cr = cv2.adaptiveThreshold(car_img_Cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value)


    return th_Cr


def MorphologicalOperations(img):
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    kernel = np.ones((9, 9), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    # cv2.imshow("Erosion", erosion)
    # cv2.imshow("Dilation", dilation)

    return dilation


def DrawBestPair(img, pair, labels):
    if len(pair) == 0:
        # cv2.imshow("Output", img)
        return

    zone_i = pair[0]
    zone_j = pair[1]



    ymax_i, xmax_i = np.max(np.where(labels == zone_i), 1)
    ymin_i, xmin_i = np.min(np.where(labels == zone_i), 1)

    ymax_j, xmax_j = np.max(np.where(labels == zone_j), 1)
    ymin_j, xmin_j = np.min(np.where(labels == zone_j), 1)
    #


    cv2.rectangle(img, (xmin_i, ymin_i), (xmax_i, ymax_i), (0, 0, 255), 2)
    cv2.rectangle(img, (xmin_j, ymin_j), (xmax_j, ymax_j), (0, 0, 255), 2)


    cv2.imshow("Output", img)

    rects = []
    rects.append([[xmin_i, ymin_i], [xmax_i, ymax_i]])
    # print("rect1 {}".format(rects))
    rects.append([[xmin_j, ymin_j], [xmax_j, ymax_j]])
    # print("rect2 {}".format(rects))
    return rects


def TailDetector(img):
    img_yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    threshold_img = GetThresholdImg(img_yCrCb)
    morpho_img = MorphologicalOperations(threshold_img)

    # cv2.imshow('m',morpho_img)

    connectivity = 4
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morpho_img, connectivity, cv2.CV_32S)

    light_pairs = SymmetryTest(morpho_img, n_labels, labels, stats, centroids)

    pair_with_max_surface = []
    max_surface_value = 0

    for pair in light_pairs:
        part_1 = pair[0]
        part_2 = pair[1]

        # print(part_1,part_2)
        surf1 = len([element for element in labels.flatten() if element == part_1])
        surf2 = len([element for element in labels.flatten() if element == part_2])

        surface_sum = surf1 + surf2

        if surface_sum > max_surface_value:
            max_surface_value = surface_sum
            pair_with_max_surface = pair
            #print(("max_surface_value {}.".format(max_surface_value)))

    best_pair_bboxes = DrawBestPair(img, pair_with_max_surface, labels)

    # print(best_pair_bboxes[0][0][1])
    return np.array(best_pair_bboxes)



# car_img = cv2.imread('./testing_data/car_stopped.png', cv2.IMREAD_COLOR)
# TailDetector(car_img)
# cv2.waitKey(0)