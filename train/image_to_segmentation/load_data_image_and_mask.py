import pickle
import cv2
import glob

images1 = glob.glob('../../dataset/segmentation_dataset/data/*.png')
images2 = glob.glob('../../dataset/segmentation_dataset/mask/*.png')

features1 = []
features2 = []

for fname1 in images1:
    image1 = cv2.imread(fname1)
    features1.append(image1)
    print(fname1)

for fname2 in images2:
    image2 = cv2.imread(fname2, 0)
    image2 = image2.reshape(80, 160, 1)
    features2.append(image2)
    print(fname2)

with open('../../model/pickle_data/image_train.p', 'wb') as f:
    pickle.dump(features1, f, protocol=4)

with open('../../model/pickle_data/mask_labels.p', 'wb') as f:
    pickle.dump(features2, f, protocol=4)