import cv2
import numpy as np
import time
import TailDetector as tl


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


def findTails(img, car_box):
    x, y, w, h = car_box[0], car_box[1], car_box[2], car_box[3]

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 100), 1)

    if x > 0 and y > 0 and w > 0 and h > 0:
        cropped_img = img[y:y + h, x:x + w]
        cropped_img, tail_lights_bboxes = tl.TailDetector(cropped_img)

        for b_ind in range(0, len(tail_lights_bboxes)):
            tail_bbox = tail_lights_bboxes[b_ind]
            cv2.rectangle(img, (x + tail_bbox[0][0], y + tail_bbox[0][1]), (x + tail_bbox[2][0], y + tail_bbox[2][1]),
                          (52, 64, 235), 1)


# reduce if want less boxes
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

    # Non-max suppression
    indx = cv2.dnn.NMSBoxes(bbox, confs, confTresh, nmsTresh)

    for i in indx:
        i = i[0]
        car_box = bbox[i]
        findTails(img, car_box)

    cv2.imshow("Image", img)


while True:
    startCyclTime = time.time()

    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    endCyclTime = time.time()
    print(f'FPS: {round(1 / (endCyclTime - startCyclTime), 2)}')

    if cv2.waitKey(33) == 13:
        break
