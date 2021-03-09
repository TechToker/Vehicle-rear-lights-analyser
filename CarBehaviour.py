import cv2
import TailDetector as tl

car_stop = cv2.imread('./testing_data/car_example_10.png', cv2.IMREAD_COLOR)
car_not_stop = cv2.imread('./testing_data/car_example_9.png', cv2.IMREAD_COLOR)


def CarBehaviour(img1,img2):

    beh = ''
    bbox_img1 = tl.TailDetector(img1)
    bbox_img2 = tl.TailDetector(img2)

    x, y, w, h =  bbox_img1[0][0][0], bbox_img1[0][0][1], bbox_img1[0][1][0],bbox_img1[0][1][1]
    right_light_img1 = img1[y:y + h, x:x + w]
    x, y, w, h = bbox_img1[1][0][0], bbox_img1[1][0][1], bbox_img1[1][1][0], bbox_img1[1][1][1]
    left_light_img1 = img1[y:y + h, x:x + w]


    x, y, w, h = bbox_img2[0][0][0], bbox_img2[0][0][1], bbox_img2[0][1][0], bbox_img2[0][1][1]
    right_light_img2 = img2[y:y + h, x:x + w]
    x, y, w, h = bbox_img2[0][0][0], bbox_img2[0][0][1], bbox_img2[0][1][0], bbox_img2[0][1][1]
    left_light_img2 = img2[y:y + h, x:x + w]

    #самый тупой способ
    status = False
    img1_yCrCb_r = cv2.cvtColor(right_light_img1, cv2.COLOR_BGR2YCrCb)
    img1_yCrCb_l = cv2.cvtColor(left_light_img1, cv2.COLOR_BGR2YCrCb)

    img2_yCrCb_r = cv2.cvtColor(right_light_img2, cv2.COLOR_BGR2YCrCb)
    img2_yCrCb_l = cv2.cvtColor(left_light_img2, cv2.COLOR_BGR2YCrCb)

    img1_Y_r,img1_Cr_r,img1_Cb_r = cv2.split(img1_yCrCb_r)
    img1_Y_l, img1_Cr_l, img1_Cb_l = cv2.split(img1_yCrCb_l)

    img2_Y_r,img2_Cr_r,img2_Cb_r = cv2.split(img2_yCrCb_r)
    img2_Y_l, img2_Cr_l, img2_Cb_l = cv2.split(img2_yCrCb_l)


    ret, thresh1_r = cv2.threshold(img1_Cr_r, 200, 255, cv2.THRESH_BINARY)
    ret, thresh1_l = cv2.threshold(img1_Cr_l, 200, 255, cv2.THRESH_BINARY)

    ret2, thresh2_r = cv2.threshold(img2_Cr_r, 200, 255, cv2.THRESH_BINARY)
    ret, thresh2_l = cv2.threshold(img2_Cr_l, 200, 255, cv2.THRESH_BINARY)

    non1_r = cv2.countNonZero(thresh1_r)
    non1_l = cv2.countNonZero(thresh1_l)

    non2_r = cv2.countNonZero(thresh2_r)
    non2_l = cv2.countNonZero(thresh2_l)

    non1 = non1_r+non1_l
    non2 = non2_r + non2_l

    difference = abs(non1 - non2)

    print(difference)

    if difference > 800:
        status = True

    if status == True:
        difference = non1 - non2
        if difference >0:
            beh = 'off'
        else:
            beh = 'on'

    return f"Is status changed -  {status}, the light are {beh}"

print(CarBehaviour(car_stop,car_not_stop))
# #
cv2.imshow("not_stop",CarBehaviour(car_not_stop,car_stop))
cv2.imshow("stop",CarBehaviour(car_stop,car_not_stop))
cv2.waitKey(0)