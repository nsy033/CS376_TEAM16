import numpy as np
import csv
from pydub import AudioSegment

f = open('./30_data_more.csv','r')
rd = csv.reader(f)
data = list(rd)
idx_col = [str(i) for i in range (len(data))]
data = np.c_[idx_col, data]

np.random.shuffle(data)

train = []
valid = []
test = []
binary = []
for i in range(len(data)):
    print(data[i])
    if i < 636:
        train.append(data[i][0] + "\t" + data[i][1])
    elif i < 736:
        valid.append(data[i][0] + "\t" + data[i][1])
    else:
        test.append(data[i][0] + "\t" + data[i][1])
    binary.append(data[i][2:])

train = np.array(train)
valid = np.array(valid)
test = np.array(test)
binary = np.array(binary)

np.save('train.npy', train)
np.save('valid.npy', valid)
np.save('test.npy', test)
np.save('binary.npy', binary)