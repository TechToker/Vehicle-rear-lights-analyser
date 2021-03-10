from matplotlib import pyplot as plt
import numpy as np
import cv2
from FramesStorage import *

# In mls
BRAKE_ANALYSIS_TIME = 150

BRAKE_ANALYSIS_TIME_1 = 300
BRAKE_ANALYSIS_TIME_2 = 500

STATUS_CHANGE_THRESHOLD = 15

def generate_colors(num):
    #r = lambda: np.random.randint(0, 255)
    #return [(r(), r(), r()) for _ in range(num)]

    return [(208, 86, 93), (197, 162, 4), (233, 43, 131), (203, 208, 223), (18, 121, 41), (64, 85, 147),
            (206, 187, 204), (36, 72, 148), (158, 11, 209), (36, 1, 154), (96, 53, 119), (230, 60, 218),
            (206, 187, 204), (36, 72, 148), (158, 11, 209), (36, 1, 154), (96, 53, 119), (230, 60, 218)]


def SymmetryTest(img, n_labels, labels, stats, centroids, test_str):
    source_img = img.copy()
    cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)

    labeled_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    light_pairs = []

    colors = generate_colors(n_labels)

    # print(img.shape)
    #
    # cent_x, cent_y = int(centroids[0, 0]), int(centroids[0, 1])
    # cv2.circle(source_img, (cent_x, cent_y), 3, (128, 128, 128), -1)
    # cv2.putText(source_img, f"{0}; {cent_x, cent_y}", (cent_x, cent_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

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
            dist_between_lights = 20

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

    #cv2.imshow(f"Image:{test_str}", labeled_image)

    return light_pairs


def GetThresholdImg(img):
    car_img_Y = img[:, :, 0]
    car_img_Cr = img[:, :, 1]
    car_img_Cb = img[:, :, 2]

    block_size = 11
    c_value = 7
    th_Y = cv2.adaptiveThreshold(car_img_Y, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value)
    th_Cr = cv2.adaptiveThreshold(car_img_Cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_value)

    # If amount of enabled pixels lower than 5 => hystogram approach
    # th_Cr_1dim = np.array(th_Cr)
    # if sum(th_Cr_1dim > 0) < 5:
    #     max_red_intensity = np.amax(car_img_Cr)
    #     lower_threshold = 10
    #
    #     ret, th_Cr = cv2.threshold(car_img_Cr, max_red_intensity - lower_threshold, 255, cv2.THRESH_BINARY)

    # cv2.imshow("Image12", car_img_Cr)
    #cv2.imshow("Image14", th_Cr)

    # hist_full = cv2.calcHist([car_img_Cr], [0], None, [256], [0, 256])
    # plt.plot(hist_full, color='r')
    # plt.show()

    return th_Cr


def MorphologicalOperations(img):
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    kernel = np.ones((9, 9), np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    # cv2.imshow("Erosion", erosion)
    #cv2.imshow("Dilation", dilation)

    return dilation


def DrawBestPair(img, pair, labels):
    if len(pair) == 0:
        #cv2.imshow("Output", img)
        return pair

    zone_i = pair[0]
    zone_j = pair[1]

    ymax_i, xmax_i = np.max(np.where(labels == zone_i), 1)
    ymin_i, xmin_i = np.min(np.where(labels == zone_i), 1)

    ymax_j, xmax_j = np.max(np.where(labels == zone_j), 1)
    ymin_j, xmin_j = np.min(np.where(labels == zone_j), 1)

    # cv2.rectangle(img, (xmin_i, ymin_i), (xmax_i, ymax_i), (0, 0, 255), 2)
    # cv2.rectangle(img, (xmin_j, ymin_j), (xmax_j, ymax_j), (0, 0, 255), 2)

    #cv2.imshow(f"Output: {xmin_j}", img)

    rects = []
    rects.append([xmin_i, ymin_i, xmax_i, ymax_i])
    rects.append([xmin_j, ymin_j, xmax_j, ymax_j])

    return rects


def TailDetector(img, IsPast):
    #TODO: Remove this
    if IsPast:
        str = "[ Past ]"
    else:
        str = "[Actual]"

    car_img_rgb = img.copy()

    img_yCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    threshold_img = GetThresholdImg(img_yCrCb)
    morpho_img = MorphologicalOperations(threshold_img)

    connectivity = 4
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morpho_img, connectivity, cv2.CV_32S)

    light_pairs = SymmetryTest(morpho_img, n_labels, labels, stats, centroids, str)

    pair_with_max_surface = []
    max_surface_value = 0
    surface_mean = 0

    for pair in light_pairs:
        part_1 = pair[0]
        part_2 = pair[1]

        flatten_labels = labels.flatten()

        surf1 = [element for element in flatten_labels if element == part_1]
        surf2 = [element for element in flatten_labels if element == part_2]

        surface_sum = len(surf1) + len(surf2)

        if surface_sum > max_surface_value:
            pair_with_max_surface = pair
            max_surface_value = surface_sum
            surface_mean = np.mean(surf1) + np.mean(surf2)

    bboxes = DrawBestPair(img, pair_with_max_surface, labels)

    total_mean_Cr = 0
    total_mean_Y = 0

    if len(pair_with_max_surface) > 0:
        test = img_yCrCb[:, :, 1].copy()

        sum1 = np.mean(test[labels == pair_with_max_surface[0]].flatten())
        sum2 = np.mean(test[labels == pair_with_max_surface[1]].flatten())
        total_mean_Cr = np.mean([sum1, sum2])

        #print(f"Total mean Cr: {total_mean_Cr};")

        test = img_yCrCb[:, :, 0].copy()

        sum1 = np.mean(test[labels == pair_with_max_surface[0]].flatten())
        sum2 = np.mean(test[labels == pair_with_max_surface[1]].flatten())
        total_mean_Y = np.mean([sum1, sum2])

        #print(f"{str} Mean Y: {round(total_mean_Y, 2)}")

    #cv2.imshow(f"Name", test)
    # TODO: Find third brake light

    return np.array(bboxes), img_yCrCb, total_mean_Y, total_mean_Cr


# ! img in YCrCb
def CalculateRednessInAreas(img, rects):
    if len(rects) == 0:
        return 0

    # print(f"r: {rects.size}")
    # print(f"r: {rects}")
    # print(f"r: {rects is None}")

    areas_mean = []
    for i in range(len(rects)):
        light_rect = rects[i]
        crop_area = img[light_rect[1]: light_rect[3], light_rect[0]: light_rect[2]][:, :, 1]

        #cv2.imshow(f"Light past{rects}", crop_area)
        areas_mean.append(np.mean(crop_area.flatten()))

    return np.mean(areas_mean)


def CompareTails(img_from_past, img_actual):

    res_past, past_img, past_mean_Y, past_mean_Cr = TailDetector(img_from_past, True)
    res_current, cur_img, cur_mean_Y, cur_mean_Cr = TailDetector(img_actual, False)

    # past_redness_mean = CalculateRednessInAreas(past_img, res_past)
    # print(f"Past car: {past_surface_mean};")
    #
    # actual_redness_mean = CalculateRednessInAreas(cur_img, res_current)
    # print(f"Actual car: {cur_surface_mean};")

    #cv2.imshow(f"Light past", past_img)
    #cv2.imshow(f"Light actual", cur_img)

    # if past_mean_Cr > 0 and cur_mean_Y > 0 and abs(past_mean_Y - cur_mean_Y) > STATUS_CHANGE_THRESHOLD:
    #     if cur_mean_Y > past_mean_Y:
    #         print(f"[NewStatus]: {CarStatus.BRAKING}; Past: {past_mean_Y} Actual: {cur_mean_Y}")
    #     else:
    #         print(f"[NewStatus]: {CarStatus.NOT_BRAKING}; Past: {past_mean_Y} Actual: {cur_mean_Y}")

    return past_mean_Y, cur_mean_Y, res_past, res_current


def AnalyzeCarStatus(current_time, car):
    prev_car_frame = car.GetFrameFromPast(current_time, BRAKE_ANALYSIS_TIME)
    if prev_car_frame is None:
        return []

    actual_frame = car.GetLastFrame().GetImage()
    past_frame = prev_car_frame.GetImage()

    # cv2.imshow("Past", past_frame)
    # cv2.imshow("Actual", actual_frame)

    past_redness, actual_redness, bb_past, bb_current = CompareTails(past_frame, actual_frame)
    car.GetLastFrame().SetBrightness(actual_redness)

    past_redness = car.GetAverageBrightness(current_time, BRAKE_ANALYSIS_TIME_1, BRAKE_ANALYSIS_TIME_2)
    actual_redness = car.GetAverageBrightness(current_time, 0, BRAKE_ANALYSIS_TIME)

    #print(past_redness)

    print(f"[Past] Mean Y: {round(past_redness, 2)}")
    print(f"[Actual] Mean Y: {round(actual_redness, 2)}")

    if abs(past_redness - actual_redness) > STATUS_CHANGE_THRESHOLD:
        #if actual_redness > 55:
        if actual_redness > past_redness:
            car.SetStatus(CarStatus.BRAKING)
            print(f"[NewStatus]: {CarStatus.BRAKING}; Past: {past_redness} Actual: {actual_redness}")
        else:
            car.SetStatus(CarStatus.NOT_BRAKING)
            print(f"[NewStatus]: {CarStatus.NOT_BRAKING}; Past: {past_redness} Actual: {actual_redness}")

    return bb_current

# Uncommit it to test it on signe image

# past_frame = cv2.imread('./testing_data/car_example_9.png', cv2.IMREAD_COLOR)
# cur_frame = cv2.imread('./testing_data/car_example_10.png', cv2.IMREAD_COLOR)
#
# CompareTails(past_frame, cur_frame)
# cv2.waitKey(0)

# car_img = cv2.imread('./testing_data/car_example_5.png', cv2.IMREAD_COLOR)
# TailDetector(car_img)
# cv2.waitKey(0)