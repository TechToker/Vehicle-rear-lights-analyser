class CarFrame:
    time = 0

    # Format - [start_y, start_x, height, width]
    bounding_box = []
    centroid = []

    image = []

    def __init__(self, frame_time, bounding_box, image):
        self.time = frame_time
        self.bounding_box = bounding_box

        center_y = bounding_box[0] + int(bounding_box[2] / 2)
        center_x = bounding_box[1] + int(bounding_box[3] / 2)

        self.centroid = [center_y, center_x]
        self.image = image

    def GetTime(self):
        return self.time

    def GetBoundingBox(self):
        return self.bounding_box

    def GetCentroid(self):
        return self.centroid

    def GetSize(self):
        return [self.bounding_box[2], self.bounding_box[3]]

    def GetImage(self):
        return self.image
