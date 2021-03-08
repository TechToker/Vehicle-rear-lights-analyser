import cv2
import numpy as np
from enum import Enum
car_stop = cv2.imread('./testing_data/car_stopped.png', cv2.IMREAD_COLOR)
car_not_stop = cv2.imread('./testing_data/car_not_stopped.png', cv2.IMREAD_COLOR)


def CarBehaviour(img1,img2):

    #самый тупой способ
    status = None
    img1_yCrCb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    img2_yCrCb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)

    img1_Y,img1_Cr,img1_Cb = cv2.split(img1_yCrCb)
    img2_Y, img2_Cr, img2_Cb = cv2.split(img2_yCrCb)

    ret, thresh1 = cv2.threshold(img1_Cr, 200, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img2_Cr, 200, 255, cv2.THRESH_BINARY)
    non1 = cv2.countNonZero(thresh1)
    non2 = cv2.countNonZero(thresh2)
    difference = abs(non1 - non2)
    print(difference)
    #не знаю как корректнее сделать пока что от балды
    if difference > 1000:
        status = 'braking'
    else:
        status ='driving'

    return status



# print(CarBehaviour(car_not_stop,car_stop))
# #
# cv2.imshow("not_stop",CarBehaviour(car_not_stop,car_stop))
# cv2.imshow("stop",CarBehaviour(car_stop,car_not_stop))
# cv2.waitKey(0)