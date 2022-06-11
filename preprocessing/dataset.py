import csv
import numpy as np

f = open('/Users/yeon/2022KAIST/1_spring/[CS376]MachineLearning/final_project/preprocessing/data.csv', 'r')
reader = csv.reader(f)
data = list(reader)
tags = data[0][1:]
binary_w_title = data[1:]
binary = []
titles = []

idx = 0
for row in binary_w_title:
    titles.append(str(idx) + "\t" + row[0])
    binary.append(row[1:])
    idx += 1

titles = np.array(titles)
tags = np.array(tags)
binary = np.array(binary)
train = np.array(titles[:80])
valid = np.array(titles[80:90])
test = np.array(titles[90:100])

np.save('tags.npy', tags)
np.save('binary.npy', binary)
np.save('train.npy', train)
np.save('valid.npy', valid)
np.save('test.npy', test)

# print(tags)
print(train)
print(valid)
print(test)
# print(len(binary[0]))
# print(len(tags))
