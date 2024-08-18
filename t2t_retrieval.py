import os
import numpy as np
from tqdm import tqdm

data_path = '../dataset/cub/attributes/image_attribute_labels.txt'
best_path = "../dataset/cub/t2t_best"
data = np.ones((11788, 312))

with open(data_path, 'r') as data_file:
    lines = data_file.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        data[int(line[0])-1][int(line[1])-1] = int(line[2])
        if line[3]==str(1):
            data[int(line[0])-1][int(line[1])-1] = -1

with open('../dataset/cub/test.txt', 'r') as tef:
    tests = tef.readlines()
with open('../dataset/cub/train.txt', 'r') as trf:
    trains = trf.readlines()

for test in tests:
    test = int(test.strip())
    best_score = -1
    best_id = -1
    for train in trains:
        train = int(train.strip())
        score = 0
        for i in range(312):
            #print(data[test-1][i], data[train-1][i], data[test-1][i] == data[train-1][i])
            if data[test-1][i] == data[train-1][i]:
                score += 1
        #print(score)
        if score > best_score:
            best_score = score
            best_id = train
    print(best_score)
    with open(os.path.join(best_path, str(test)+".txt"), "w") as f:
        f.write(str(best_id))
