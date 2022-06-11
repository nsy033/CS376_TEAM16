from pydub import AudioSegment
import numpy as np
import csv

f = open('./data.csv', 'r')
reader = csv.reader(f)
data = list(reader)
binary_w_title = data[1:]
titles = []

idx = 0
for row in binary_w_title:
    titles.append(row[0])

for title in titles:
    # Open file
    song = AudioSegment.from_file("/Users/yeon/2022KAIST/1_spring/[CS376]MachineLearning/final_project/preprocessing/medium/" + title, "mp3")

    for i in range(3):
        # Slice audio
        # pydub는 milliseconds 단위를 사용한다
        # sliced = song[:29100]
        if len(song) < (i+1)*29100:
            break

        sliced = song[i*29100: (i+1)*29100]

        # Save the result
        # can give parameters-quality, channel, etc
        sliced.export((title + "_" + str(i * 29) + "_" + str((i+1) * 29)).replace('.mp3', '') + '.wav', format='wav')
    