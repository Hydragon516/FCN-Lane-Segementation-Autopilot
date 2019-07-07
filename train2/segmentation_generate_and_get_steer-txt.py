import cv2
import numpy as np
from scipy.misc import imresize
from keras.models import load_model
import pickle

model1 = load_model('../model/model_files/image_to_segmentation_lane.h5')


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

if __name__ == '__main__':

    cap = cv2.VideoCapture("video2.mp4")

    i = 0
    j = 0
    features = []
    steer = []

    while True:
        _, img1 = cap.read()
        result = road_lines(img1)

        result = result.reshape(80, 160, 1)
        features.append(result)

        cv2.imshow('result', result)
        cv2.imshow('frame', img1)

        i = i + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open('../model/pickle_files/segmentation_train.p', 'wb') as f:
                pickle.dump(features, f, protocol=4)
            with open("C:/Users/nicet/Desktop/data/images/data/data2.txt") as f:
                for line in f:
                    split_data = line.replace(',', ' ').split()
                    print(split_data)
                    steer.append(float(split_data[1]))
                    j = j + 1
                    if j >= i:
                        break
            with open('../model/pickle_files/steer_labels.p', 'wb') as fs:
                pickle.dump(steer, fs, protocol=4)
            print('save complete')
            break