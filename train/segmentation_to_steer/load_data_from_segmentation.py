import os
import natsort
import cv2
import pickle

features = []

path_dir = '../../dataset/steering_dataset/segmentation'
file_list = os.listdir(path_dir)
file_list = [file for file in file_list if file.endswith(".jpg")]
file_list = natsort.natsorted(file_list)

for i in file_list:
    print(i)
    image = cv2.imread('../../dataset/steering_dataset/segmentation/' + str(i), 0)
    image = image.reshape(80, 160, 1)
    features.append(image)

with open('../../model/pickle_data/segmentation_train.p', 'wb') as f:
    pickle.dump(features, f, protocol=4)