import cv2
import numpy as np
import TailDetector as tl
import time

from FramesStorage import *

# Settings
IS_FPS_SHOW = False
BOUNDING_BOXES_LIMIT = 0 # if zero => unlimited
CAR_ID_TO_PROCESS = [2] # [0] #[1] # [0, 1, 7] # Process only the car with this ids; Empty list - process all

# Settings for saving video
cap = cv2.VideoCapture('./testing_data/road_2.mp4')

(grabbed, frame) = cap.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./testing_data/_resulting_video.mp4', fourcc, 20.0, (fwidth, fheight))

# YOLO params
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


def GenerateColors(num):
    r = lambda: np.random.randint(64, 255)
    return [(r(), r(), r()) for _ in range(num)]


color_palette = GenerateColors(30)


# TODO: Convert in into seconds
def GetCurrentTimestamp():
    return cap.get(cv2.CAP_PROP_POS_MSEC)


def TailsProcessing(car):
    rects = tl.AnalyzeCarStatus(GetCurrentTimestamp(), car)
    return rects


# TODO: Find more clever solution; (also it is duplicate of method from TailDetector)
def GetColor(id):
    return np.array(color_palette[id])


# TODO: Move it to helper method
def DrawCarPath(source_img, path, car_id):
    # Loop from old to new points
    for ind in range(len(path)):
        color = GetColor(car_id)

        # TODO: FIX THAT
        # TODO: Transparent color
        color[0] -= (len(path) - ind) * 3
        color[1] -= (len(path) - ind) * 3
        color[2] -= (len(path) - ind) * 3

        point = path[ind]

        # Circle size changing
        circle_size = int(9 - (len(path) - ind) * 0.1)

        cv2.circle(source_img, (int(point[0]), int(point[1])), circle_size, (int(color[0]), int(color[1]), int(color[2])), -1)


def BoundingBoxProcessing(source_img, bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    if x <= 0 or y <= 0 or w == 0 or h == 0:
        return source_img

    cropped_img = source_img[y:y + h, x:x + w].copy()

    timestamp = GetCurrentTimestamp()

    frameStorage.ClearLongTimeUndetectableCars(timestamp)
    frameStorage.ClearOldFrames(timestamp)

    current_car = frameStorage.GetCar(timestamp, bbox, cropped_img)
    car_id = current_car.GetId()

    if len(CAR_ID_TO_PROCESS) != 0 and car_id not in CAR_ID_TO_PROCESS:
        return source_img

    # Tracked path
    path = frameStorage.GetCarPath(car_id)
    #DrawCarPath(source_img, path, car_id)

    # Tails
    rects = TailsProcessing(current_car)

    if len(rects) == 0:
        return

    color = [0, 0, 0]
    if current_car.GetStatus() == CarStatus.BRAKING:
        color = [0, 0, 255]
    elif current_car.GetStatus() == CarStatus.NOT_BRAKING:
        color = [0, 0, 128]
    else:
        color = [0, 0, 0]

    for i in range(len(rects)):
        rect = rects[i]
        cv2.rectangle(source_img, (x + rect[0], y + rect[1]), (x + rect[2], y + rect[3]), color, 2)

    # Visualisation
    cv2.rectangle(source_img, (x, y), (x + w, y + h), (52, 64, 235), 1)
    cv2.putText(source_img, f"Id: {car_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 235), 2)
    #cv2.putText(source_img, f"{current_car.GetStatus()}", (x + 50, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 235), 2)

    return source_img

def GetDetections(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)

            if 0 < classId < 7:
                confidence = scores[classId]
                if confidence > confTresh:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)

                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confidences.append(float(confidence))

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(bbox, confidences, confTresh, nmsTresh)

    bounding_boxes_amount = BOUNDING_BOXES_LIMIT if BOUNDING_BOXES_LIMIT > 0 else len(indexes)

    for i in range(bounding_boxes_amount):
        ind = indexes[i][0]
        car_bbox = bbox[ind]
        res_img = BoundingBoxProcessing(img, car_bbox)

    cv2.imshow("Image", img)
    out.write(img)


while True:
    startCycleTime = time.time()
    success, frame_img = cap.read()

    if not success:
        break

    blob = cv2.dnn.blobFromImage(frame_img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(outputNames)
    GetDetections(detections, frame_img)

    endCycleTime = time.time()

    if IS_FPS_SHOW:
        print(f'FPS: {round(1 / (endCycleTime - startCycleTime), 2)}')

    if cv2.waitKey(33) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
