import cv2
import numpy as np
import time
import sys

cap = cv2.VideoCapture('./testing_data/road_8sec.mp4')

classesFile = './Yolo_conf/coco.names'
modelConfiguration = './Yolo_conf/yolov3.cfg'
modelWeights = './Yolo_conf/yolov3.weights'

whT = 320

classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(classNames)

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confTresh = 0.5
nmsTresh = 0.3


# reduce if want less boxes

def TailDetector(img):
    car_img_rgb = img.copy() #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

        print(xmin, ymin, xmax, ymax)

        bboxes.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        cv2.rectangle(car_img_rgb, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 0, 0), 2)

    return car_img_rgb, np.array(bboxes)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confTresh:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # print(len(bbox))

    # Non-max suppression
    indx = cv2.dnn.NMSBoxes(bbox, confs, confTresh, nmsTresh)

    test_box = bbox[0]
    x, y, w, h = test_box[0], test_box[1], test_box[2], test_box[3]
    if x > 0 and y > 0 and w > 0 and h > 0:
        cropped_img = img[y:y + h, x:x + w]

        print(x, y, w, h)

        cropped_img, tail_lights_bboxes = TailDetector(cropped_img)
        cv2.imshow("Image", cropped_img)

    # for i in indx:
    #     i = i[0]
    #     box = bbox[i]
    #     x, y, w, h = box[0], box[1], box[2], box[3]
    #
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (242, 136, 208), 2)
    #
    #     # cv2.putText(img,f'{round(confs[i],2)}',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    #     cv2.putText(img, f'{classNames[classIds[i]]}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)



    # for i in tail_lights_bboxes:
    #     print(tail_lights_bboxes)
    #     box = tail_lights_bboxes[i]
    #     x, y, w, h = box[0], box[1], box[2], box[3]
    #     if x > 0 and y > 0 and w > 0 and h > 0:
    #         #cv2.rectangle(img, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 0, 0), 2)
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #cv2.imshow("Image", img)

while True:
    startCyclTime = time.time()

    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    # print(net.getUnconnectedOutLayers())

    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # 300 bound boxes prod

    findObjects(outputs, img)

    endCyclTime = time.time()
    print(f'FPS: {round(1 / (endCyclTime - startCyclTime), 2)}')

    #cv2.imshow("Image", img)

    if cv2.waitKey(33) == 13:
        break
