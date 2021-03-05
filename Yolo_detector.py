import cv2
import numpy as np
import TailDetector as tl


cap = cv2.VideoCapture('./testing_data/road_8sec.mp4')

#settings for saving video
(grabbed, frame) = cap.read()
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./testing_data/road_8sec_result.mp4',fourcc, 20.0, (fwidth,fheight))

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
    if x <= 0 or y <= 0 or w == 0 or h == 0:
        return

    cropped_img = img[y:y + h, x:x + w]
    tail_lights_bboxes = tl.TailDetector(cropped_img)

    # WTF??? (Need to check that list not empty)
    if tail_lights_bboxes.size == 1:
        return

    for bbox in tail_lights_bboxes:
        cv2.rectangle(img, (x + bbox[0][0], y + bbox[0][1]), (x + bbox[1][0], y + bbox[1][1]), (52, 64, 235), 1)


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
        findTails(img, car_box)

        # x, y, w, h = car_box[0], car_box[1], car_box[2], car_box[3]
        # img = img[y:y + h, x:x + w]

    cv2.imshow("Image", img)
    out.write(img)


while True:

    success, img = cap.read()

    if not success:
        break

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    if cv2.waitKey(33) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()