import cv2
import os
import natsort

img_array = []

path_dir = 'D:/dataset/dataset/steering/driving_dataset/'
file_list = os.listdir(path_dir)
file_list = [file for file in file_list if file.endswith(".jpg")]
file_list = natsort.natsorted(file_list)

for filename in file_list:
    print(filename)
    img = cv2.imread('D:/dataset/dataset/steering/driving_dataset/'+filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('../test_video/video2.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
    print(i)
out.release()