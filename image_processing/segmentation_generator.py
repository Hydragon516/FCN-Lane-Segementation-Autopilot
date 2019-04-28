import numpy as np
import cv2
from scipy.misc import imresize
from keras.models import load_model
import os
import natsort

model1 = load_model('../model/model_files/image_to_segmentation.h5')

path_dir = '../dataset/steering_dataset/data'
file_list = os.listdir(path_dir)
file_list = [file for file in file_list if file.endswith(".jpg")]
file_list = natsort.natsorted(file_list)


class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction1 = model1.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction1)

    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    test = lanes.avg_fit.astype(np.uint8)

    return test

lanes = Lanes()

for i in file_list:
    print(i)
    frame = cv2.imread('../dataset/steering_dataset/data/' + str(i), cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
    result = road_lines(frame)

    cv2.imwrite('../dataset/steering_dataset/segmentation/' + str(i), result)