class DetectedCar:
    id = 0
    frames = []

    def __init__(self, car_id, first_frame):
        self.id = car_id
        self.frames = [first_frame]

    def GetId(self):
        return self.id

    def AddFrame(self, frame):
        self.frames.append(frame)

    def GetLastFrame(self):
        return self.frames[len(self.frames) - 1]
