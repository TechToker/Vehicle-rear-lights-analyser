from DetectedCar import *
from CarFrame import *

class FramesStorage:
    detected_cars = []

    def __init__(self):
        self.detected_cars = []

    def GetCarId(self, time, bounding_box, crop_img):

        return time

        car_frame = CarFrame(time, bounding_box, crop_img)

        # TODO: Search for all detected cars last frames and find near bounding box
        car_id = 0

        # If this is first time when we see that car => write it to array
        isFirstTime = False

        if(isFirstTime):
            detected_car = DetectedCar(car_id, car_frame)
            self.detected_cars.append(detected_car)
        else:
            current_car = None
            #self.detected_cars[current_car]

        # TODO: Write to array detected_cars


        return time