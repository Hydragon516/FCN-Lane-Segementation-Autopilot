import pickle

steer = []

start_num = 0 # your start image number
end_num = 40702 # your last image number

train_batch_pointer = 0
val_batch_pointer = 0

with open("../../dataset/steering_dataset/steer_data/data.txt") as f:
    for line in f:
        start_num = start_num + 1
        split_data = line.replace(',', ' ').split()
        print(split_data[1])
        steer.append(split_data[1])

        if start_num > end_num:
            break

with open('../../model/pickle_data/steer_labels.p', 'wb') as fs:
    pickle.dump(steer, fs, protocol=4)