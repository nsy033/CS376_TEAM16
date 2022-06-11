import numpy as np
import csv
from pydub import AudioSegment

f = open('data.csv','r')
f2 = open('new_data.csv','w',newline='')
rd = csv.reader(f)
data = list(rd)
wr = csv.writer(f2)
idx = 0
binary = np.load('./binary.npy')

for j in range(300):
    title = data[j][0]
    print(j, "\t", "/Users/yeon/Desktop/music/" + title)
    song = AudioSegment.from_file("/Users/yeon/Desktop/music/" + title)

    for i in range(3):
        # Slice audio
        # pydub는 milliseconds 단위를 사용한다
        # sliced = song[:29100]
        if len(song) < (i+1)*29100:
            break
        
        if i == 0:
            wr.writerow(data[j])
        else:
            sliced = song[i*29100: (i+1)*29100]

            # Save the result
            # can give parameters-quality, channel, etc
            sliced.export((title + "_" + str(i * 29) + "_" + str((i+1) * 29)).replace('.mp3', '') + '.wav', format='wav')

            input = np.append((title + "_" + str(i * 29) + "_" + str((i+1) * 29)).replace('.mp3', '') + '.mp3', binary[j])
            wr.writerow(input)