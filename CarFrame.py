class CarFrame:
    time = 0

    # Format - [start_y, start_x, height, width]
    bounding_box = []
    frame = []

    def __init__(self, frame_time, bounding_box, frame):
        self.time = frame_time
        self.bounding_box = bounding_box
        self.frame = frame

    def GetTime(self):
        return self.time

    def GetBoundingBox(self):
        return self.bounding_box

    def GetFrame(self):
        return self.frame
