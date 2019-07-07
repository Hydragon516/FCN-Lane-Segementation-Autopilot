import cv2
import numpy as np
from scipy.misc import imresize
from keras.models import load_model
import pickle

model1 = load_model('../model/model_files/image_to_segmentation.h5')

class OptFlow:
    def __init__(self, resize_width=320, resize_height=180, height_start=0.2, height_end=0.5):
        self.width = resize_width
        self.height = resize_height

        self.height_start = int(self.height * height_start)
        self.height_end = int(self.height * height_end)

    def get_direction(self, frame1, frame2, show=False):
        frame1 = cv2.resize(frame1, (self.width, self.height))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2, (self.width, self.height))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1[self.height_start:self.height_end],
                                            frame2[self.height_start:self.height_end], None, 0.5, 3, 15, 1, 5, 1.2, 0)
        flow_avg = np.median(flow, axis=(0, 1))  # [x, y]

        move_x = -1 * flow_avg[0]
        move_y = -1 * flow_avg[1]

        if show:
            hsv = np.zeros((self.height_end - self.height_start, self.width, 3))
            hsv[...,1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(np.array(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

            cv2.imshow('opt_flow', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                exit(1)

        return move_x, move_y

    @staticmethod
    def draw_arrow(img, x, y, multiply=100):
        h, w, c = img.shape
        arrow = cv2.arrowedLine(img, (int(w / 2), int(h / 2)), (int(w / 2 + x * multiply), int(h / 2)),
                                color=(0, 255, 255), thickness=15)
        return arrow


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
    flow = OptFlow()

    cap = cv2.VideoCapture("video.mp4")
    _, img1 = cap.read()
    _, img2 = cap.read()

    i = 0
    features = []
    steer = []

    while True:
        x, y = flow.get_direction(img1, img2)
        STEER = x*50
        print('STEER: {}'.format(x*50))

        arrow = flow.draw_arrow(img1, x, y)
        arrow = cv2.resize(arrow, dsize=(640, 360), interpolation=cv2.INTER_AREA)

        result = road_lines(img1)

        result = result.reshape(80, 160, 1)
        features.append(result)
        steer.append(float(STEER))


        cv2.imshow('arrow', arrow)
        cv2.imshow('result', result)

        i = i + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open('../model/pickle_files/segmentation_train.p', 'wb') as f:
                pickle.dump(features, f, protocol=4)
            with open('../model/pickle_files/steer_labels.p', 'wb') as fs:
                pickle.dump(steer, fs, protocol=4)
            print('save complete')
            break

        img1 = img2
        _, img2 = cap.read()
        img2 = cv2.resize(img2, dsize=(640, 360), interpolation=cv2.INTER_AREA)