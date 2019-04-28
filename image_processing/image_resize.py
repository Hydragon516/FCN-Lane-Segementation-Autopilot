import cv2
import glob

i = 0
images = glob.glob('../dataset/segmentation_dataset/image/*.jpg')

for fname in images:
    image = cv2.imread(fname)
    image = cv2.resize(image, dsize=(160, 80), interpolation=cv2.INTER_AREA)

    print(fname)

    cv2.imwrite('../dataset/segmentation_dataset/image/' + str(i) + '.png', image)

    i = i + 1