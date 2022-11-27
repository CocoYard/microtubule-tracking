import numpy
import numpy as np
from PIL import Image


class TiffLoader:
    def __init__(self, video: np.array):
        self.tiff_video = video
        print(type(self.tiff_video))
        self.tiff_frames_num = len(video)
        self.tiff_image = []
        self.tiff_gray_image = []

        ## extract each frame from the tiff video
        for i in range(self.tiff_frames_num):
            temp = video[i]
            self.tiff_image.append(temp)
            temp = 255*((temp-temp.min())/temp.max()-temp.min())
            self.tiff_gray_image.append(temp)
        self.tiff_image = np.array(self.tiff_image)