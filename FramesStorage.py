from DetectedCar import *
from CarFrame import *
import math as math

boxes_distance_threshold = 30
boxes_size_threshold = [30, 30]


# After what time remove undetectable car from list
# In mls
undetectableCarTimeToLive = 100

class FramesStorage:
    detected_cars = []
    last_car_index = 0

    def __init__(self):
        self.detected_cars = []

    def AddNewCarToList(self, car_frame):
        car_id = self.last_car_index
        self.last_car_index += 1

        detectedCar = DetectedCar(car_id, car_frame)

        self.detected_cars.append(detectedCar)

        print(f"Find new car! Id: {car_id}")

        return car_id

    def ClearLongTimeUndetectableCars(self, current_time):

        # Reverse loop
        for i in range(len(self.detected_cars), 0, -1):
            cur_ind = i - 1

            car = self.detected_cars[cur_ind]
            lastCarFrame = car.GetLastFrame()

            # print(f"TTR: {car.GetId()}; {lastCarFrame.GetTime()}; cur time: {current_time}")
            if lastCarFrame.GetTime() + undetectableCarTimeToLive < current_time:
                print(f"Remove car: {car.GetId()}; {lastCarFrame.GetTime()}")

                self.detected_cars.pop(cur_ind)


    def GetCarId(self, time, bounding_box, crop_img):

        car_frame = CarFrame(time, bounding_box, crop_img)

        car_id = 0
        newBoxCenter = [bounding_box[0] + (int(bounding_box[2] - bounding_box[0]) / 2), bounding_box[1] + (int(bounding_box[3] - bounding_box[1]) / 2)]

        if len(self.detected_cars) > 0:

            nearDetectedCar = None
            nearCarDistance = 999

            for i in range(len(self.detected_cars)):
                car = self.detected_cars[i]

                lastBox = car.GetLastFrame().GetBoundingBox()
                lastCarCenter = [lastBox[0] + (int(lastBox[2] - lastBox[0]) / 2), lastBox[1] + (int(lastBox[3] - lastBox[1]) / 2)]

                distanceToCenter = math.dist(lastCarCenter, newBoxCenter)

                #print(f"LastBox: {lastCarCenter}; NewBox: {newBoxCenter}; Distance: {distanceToCenter}")

                if distanceToCenter < nearCarDistance:
                    nearDetectedCar = car
                    nearCarDistance = distanceToCenter

            if nearCarDistance < boxes_distance_threshold:
                car_id = nearDetectedCar.GetId()
                nearDetectedCar.AddFrame(car_frame)
            else:
                car_id = self.AddNewCarToList(car_frame)
        else:
            car_id = self.AddNewCarToList(car_frame)

        return car_id
