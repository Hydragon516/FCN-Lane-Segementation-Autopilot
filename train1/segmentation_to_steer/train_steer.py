import pickle
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

train_images = pickle.load(open("../../model/pickle_files/segmentation_train.p", "rb"))
labels = pickle.load(open("../../model/pickle_files/steer_labels.p", "rb"))

train_images = np.array(train_images)
labels = np.array(labels)

train_images, labels = shuffle(train_images, labels)

x_train, x_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

input_shape = x_train.shape[1:]


model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

early_stopping = EarlyStopping(patience=10)

model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_data=(x_val, y_val), callbacks=[early_stopping])

model.save('../../model/model_files/segmentation_to_steer.h5')