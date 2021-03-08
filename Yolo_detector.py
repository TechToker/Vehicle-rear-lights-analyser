import cv2
import numpy as np
import TailDetector as tl
import time

from FramesStorage import *

# Settings
IS_FPS_SHOW = False


cap = cv2.VideoCapture('./testing_data/road_5.mp4')

# Settings for saving video
(grabbed, frame) = cap.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./testing_data/_resulting_video.mp4', fourcc, 20.0, (fwidth, fheight))

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

frameStorage = FramesStorage()


def findTails(img, cropped_img):
    tail_lights_bboxes = tl.TailDetector(cropped_img)

    # WTF??? (Need to check that list not empty)
    if tail_lights_bboxes.size == 1:
        return

    # for bbox in tail_lights_bboxes:
    #     cv2.rectangle(img, (x + bbox[0][0], y + bbox[0][1]), (x + bbox[1][0], y + bbox[1][1]), (52, 64, 235), 1)

    return img


# Duplicate of method from TailDetector
# TODO: Find more clever solution
def GetColor(id):
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [64, 64, 64], [255, 255, 0],
              [255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [64, 64, 64], [255, 255, 0]]
    return colors[0]


def DrawCarPath(source_img, path, car_id):
    # Loop from old to new points
    for ind in range(len(path)):
        color = GetColor(car_id)

        # TODO: FIX THAT
        color[0] -= (len(path) - ind) * 5
        color[1] -= (len(path) - ind) * 5
        color[2] -= (len(path) - ind) * 5

        point = path[ind]
        cv2.circle(source_img, (int(point[0]), int(point[1])), 3, color, -1)


def BoundingBoxProcessing(source_img, bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    if x <= 0 or y <= 0 or w == 0 or h == 0:
        return

    cropped_img = img[y:y + h, x:x + w]

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    frameStorage.ClearLongTimeUndetectableCars(timestamp)
    frameStorage.ClearOldFrames(timestamp)

    car_id = frameStorage.GetCarId(timestamp, bbox, cropped_img)

    path = frameStorage.GetCarPath(car_id)
    DrawCarPath(source_img, path, car_id)

    findTails(source_img, cropped_img)

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    cv2.rectangle(source_img, (x, y), (x + w, y + h), (52, 64, 235), 1)

    cv2.putText(source_img, f"Id: {car_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 235), 2)


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
            if (classId > 0 and classId < 7):
                confidence = scores[classId]
                if confidence > confTresh:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

    # Non-max suppression
    indx = cv2.dnn.NMSBoxes(bbox, confs, confTresh, nmsTresh)

    #for i in range(0, 1):
    for i in range(0, len(indx)):
        ind = indx[i][0]
        car_box = bbox[ind]
        BoundingBoxProcessing(img, car_box)

    cv2.imshow("Image", img)
    out.write(img)


while True:
    startCyclTime = time.time()
    success, img = cap.read()

    if not success:
        break

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    endCyclTime = time.time()

    if IS_FPS_SHOW:
        print(f'FPS: {round(1 / (endCyclTime - startCyclTime), 2)}')

    if cv2.waitKey(33) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()