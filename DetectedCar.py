# In seconds
FRAME_LIFETIME = 2000


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

    def GetFirstFrame(self):
        return self.frames[0]

    def GetAllFrames(self):
        return self.frames

    def RemoveOldFrames(self, current_time):
        cleaned_list = [frame for frame in self.frames if frame.GetTime() + FRAME_LIFETIME >= current_time]
        #print(f"[Remove old frames] Car {self.id}; amount: {len(self.frames) - len(cleaned_list)}")
        self.frames = cleaned_list
