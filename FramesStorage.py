from DetectedCar import *
from CarFrame import *
import math as math
import cv2

boxes_distance_threshold = 200

# TODO: USE IT !
boxes_size_threshold = [100, 100]

# After what time remove undetectable car from list
# In milliseconds
undetectableCarTimeToLive = 500


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

        return detectedCar

    def ClearOldFrames(self, current_time):
        for car in self.detected_cars:
            car.RemoveOldFrames(current_time)

    def ClearLongTimeUndetectableCars(self, current_time):

        # Reverse loop
        for i in range(len(self.detected_cars), 0, -1):
            cur_ind = i - 1

            car = self.detected_cars[cur_ind]
            lastCarFrame = car.GetLastFrame()

            if lastCarFrame.GetTime() + undetectableCarTimeToLive < current_time:
                print(f"Remove car {car.GetId()}; CurrentTime:{current_time}, LastDetected: {lastCarFrame.GetTime()}")

                self.detected_cars.pop(cur_ind)

    def GetCar(self, time, bounding_box, crop_img):
        car_frame = CarFrame(time, bounding_box, crop_img)
        newBoxCenter = [bounding_box[0] + int(bounding_box[2] / 2), bounding_box[1] + int(bounding_box[3] / 2)]
        newBoxSize = [bounding_box[2], bounding_box[3]]

        if len(self.detected_cars) > 0:

            nearDetectedCar = None
            nearCarDistance = 999

            for i in range(len(self.detected_cars)):
                car = self.detected_cars[i]

                # Bounding box size test
                lastFrameSize = car.GetLastFrame().GetSize()

                if abs(newBoxSize[0] - lastFrameSize[0]) > boxes_size_threshold[0] or abs(newBoxSize[1] - lastFrameSize[1]) > boxes_size_threshold[1]:
                    #print("Bounding box fail")
                    # Bounding box size check fail
                    continue

                # Centroid check
                lastFrameCentroid = car.GetLastFrame().GetCentroid()
                distanceToCenter = math.dist(lastFrameCentroid, newBoxCenter)

                if distanceToCenter < nearCarDistance:
                    nearDetectedCar = car
                    nearCarDistance = distanceToCenter

            # TODO: Also check that this car is not used already!
            if nearCarDistance < boxes_distance_threshold:
                detectedCar = nearDetectedCar
                nearDetectedCar.AddFrame(car_frame)
            else:
                detectedCar = self.AddNewCarToList(car_frame)
        else:
            detectedCar = self.AddNewCarToList(car_frame)

        return detectedCar

    # TODO: Move it to Detected car method
    def GetCarPath(self, carId):
        car = next((car for car in self.detected_cars if car.GetId() == carId), None)

        all_car_frames = car.GetAllFrames()
        all_centroids = []

        for frame in all_car_frames:
            all_centroids.append(frame.GetCentroid())

        return all_centroids


