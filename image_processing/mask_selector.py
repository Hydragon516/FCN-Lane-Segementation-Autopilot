import numpy as np
import cv2
import glob

i = 0
images = glob.glob('../dataset/segmentation_dataset/label/*.png')

for fname in images:
    image = cv2.imread(fname)

    lower = np.array([0, 0, 254], dtype="uint8") #Change this value
    upper = np.array([0, 0, 255], dtype="uint8") #Change this value

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    mask = cv2.resize(mask, dsize=(160, 80), interpolation=cv2.INTER_AREA)

    print(fname)

    cv2.imwrite('../dataset/segmentation_dataset/mask/' + str(i) + '.png', mask)

    i = i + 1