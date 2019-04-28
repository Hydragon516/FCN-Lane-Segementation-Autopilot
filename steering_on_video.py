import numpy as np
import cv2
from scipy.misc import imresize
from keras.models import load_model

model1 = load_model('model/model_files/image_to_segmentation.h5')
model2 = load_model('model/model_files/segmentation_to_steer.h5')

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
    test = test[None, :]

    prediction2 = model2.predict(test)
    prediction2 = str(prediction2)
    prediction2 = prediction2[:-2]
    prediction2 = prediction2[2:]
    prediction2 = float(prediction2)
    print(prediction2)

    wheel = cv2.imread('steering_wheel.jpg')
    wheel = cv2.resize(wheel, dsize=(500, 335), interpolation=cv2.INTER_AREA)

    M = cv2.getRotationMatrix2D((500 / 2, 335 / 2), -int(prediction2), 1)

    wheel = cv2.warpAffine(wheel, M, (500, 335))

    cv2.imshow('wheel', wheel)

    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_image = imresize(lane_drawn, (360, 640, 3))

    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

lanes = Lanes()

cap = cv2.VideoCapture('test_video/lane.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
    result = road_lines(frame)

    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()