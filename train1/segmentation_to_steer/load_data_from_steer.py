import pickle

steer = []

with open("D:/dataset/dataset/steering/driving_dataset/data.txt") as f:
    for line in f:
        split_data = line.replace(',', ' ').split()
        #print(split_data)
        print(line)
        steer.append(float(split_data[1]))

with open('../../model/pickle_files/steer_labels.p', 'wb') as fs:
    pickle.dump(steer, fs, protocol=4)