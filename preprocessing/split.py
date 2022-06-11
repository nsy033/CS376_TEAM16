import numpy as np
import csv

f = open('./data.csv','r')
rd = csv.reader(f)
data = list(rd)

binary = np.load('./30_binary.npy')
# train = np.load('./train.npy')
# valid = np.load('./valid.npy')
# test = np.load('./test.npy')

fw = open('30_data.csv','w', newline='')
wr = csv.writer(fw)
for i in range(len(data)):
    print(data[i][0], len(binary[i]))
    input = np.append(data[i][0], binary[i])
    print(input)
    wr.writerow(input)
 
f.close()
