import cv2
import TailDetector as tl

car_stop = cv2.imread('./testing_data/car_stopped.png', cv2.IMREAD_COLOR)
car_not_stop = cv2.imread('./testing_data/car_not_stopped.png', cv2.IMREAD_COLOR)


def CarBehaviour(img1,img2):

    bbox_rlight = tl.TailDetector(img1)
    bbox_rlight_2 = tl.TailDetector(img2)

    x, y, w, h =  bbox_rlight[0][0][0], bbox_rlight[0][0][1], bbox_rlight[0][1][0],bbox_rlight[0][1][1]
    right_light_img_1 = img1[y:y + h, x:x + w]
    cv2.imshow('right-img1',right_light_img_1)


    x, y, w, h = bbox_rlight_2[0][0][0], bbox_rlight_2[0][0][1], bbox_rlight_2[0][1][0], bbox_rlight_2[0][1][1]
    left_light_img_1 = img2[y:y + h, x:x + w]
    cv2.imshow('right-img2', left_light_img_1)
    #самый тупой способ
    status = None
    img1_yCrCb = cv2.cvtColor(right_light_img_1, cv2.COLOR_BGR2YCrCb)
    img2_yCrCb = cv2.cvtColor(left_light_img_1, cv2.COLOR_BGR2YCrCb)

    img1_Y,img1_Cr,img1_Cb = cv2.split(img1_yCrCb)
    img2_Y, img2_Cr, img2_Cb = cv2.split(img2_yCrCb)

    ret, thresh1 = cv2.threshold(img1_Cr, 200, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img2_Cr, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('thres1-img1', thresh1)
    cv2.imshow('thres2-img2', thresh2)
    non1 = cv2.countNonZero(thresh1)
    non2 = cv2.countNonZero(thresh2)
    difference = abs(non1 - non2)

    print(difference)
    #не знаю как корректнее сделать пока что от балды
    if difference > 200:
        status = 'braking'
    else:
        status ='driving'

    return status
#
print(CarBehaviour(car_stop,car_not_stop))
# #
# cv2.imshow("not_stop",CarBehaviour(car_not_stop,car_stop))
# cv2.imshow("stop",CarBehaviour(car_stop,car_not_stop))
cv2.waitKey(0)