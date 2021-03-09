import enum

# In seconds
FRAME_LIFETIME = 2000


class CarStatus(enum.Enum):
    UNKNOWN = 0
    NOT_BRAKING = 1
    BRAKING = 2


class DetectedCar:
    id = 0
    frames = []
    status = CarStatus.UNKNOWN

    def __init__(self, car_id, first_frame):
        self.id = car_id
        self.frames = [first_frame]

    def GetId(self):
        return self.id

    def SetStatus(self, newStatus):
        self.status = newStatus

    def GetStatus(self):
        return self.status

    def AddFrame(self, frame):
        self.frames.append(frame)

    def GetLastFrame(self):
        return self.frames[len(self.frames) - 1]

    def GetFirstFrame(self):
        return self.frames[0]

    def GetAllFrames(self):
        return self.frames

    def RemoveOldFrames(self, current_time):
        cleaned_list = [frame for frame in self.frames if frame.GetTime() + FRAME_LIFETIME >= current_time]
        self.frames = cleaned_list

    def GetFrameFromPast(self, current_time, time_ago):
        # Reverse loop
        for i in range(len(self.frames) - 1, -1, -1):
            if self.frames[i].GetTime() < current_time - time_ago:
                return self.frames[i]
