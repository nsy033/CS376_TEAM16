# Music mood tagging model with multi label classification using music feature data and pre-labeled data

### Team 16 Members

	20190432 윤태양 Taeyang Yoon

	20190606 정재령 Jaeryung Chung

	20190656 최승연 Seungyeon Choi

___
### *Motivation*

Music tagging is important. With countless music being produced, it is impossible to explore the whole ocean of songs to find the right music you want. Thus, music recommendation based on music tagging is necessary for the user to make his/her own playlist. While objective features – tempo, pitch, instrument – are relatively easier to extract directly from music, subjective features – mood – are not something that could be analyzed directly from sound-wave itself. 


___
### ***Goal***

Thus, our goal is to (1) process music data and (2) build few models (CRNN and Musicnn model) that produce multiple labels of mood and (3) compare among the models to find the best model with the highest accuracy of tagging. Also, we will (4) conduct machine learning experiments with different data sets and also the parameters and the architecture. With our model, users could get better recommendation of music and listen to the right song that fits the atmosphere.

___
### ***Preprocessing***
- Library
	- We used ***librosa library*** and ***pydub library*** to preprocess mp3(wav) data and convert it into npy format. ***Librosa library*** is one of the most widely-used library for processing audio data. 
	We specifically used 

	```jsx
	librosa.core.load
	```

	to load an audio file as a floating point time series(.npy).


- Tag adjustment
    - The ***audionautix data*** had a total of 46 tags, but there were repeated tags and tags that never appear in any of the songs. So we got rid of the unneccessary tags and repeated ones, and reduces the total number of tags to 30. Also, with fma data set, we have manually labeled the data.


___
### ***Available Models***
- **Musicnn** : End-to-end Learning for Music Audio Tagging at Scale, Pons et al., 2018 [[arxiv](https://arxiv.org/abs/1711.02520)]
- **CRNN** : Convolutional Recurrent Neural Networks for Music Classification, Choi et al., 2016 [[arxiv](https://arxiv.org/abs/1609.04243)]

___
### ***Evaluation***
- Evaluation metrics we used are as follow. (TP, TN, FP, and FN refers to true positive, true negative, false positive, and false negative, respectively.)
    - Accuracy : (TP+TN) / (TP+FP+TN+FN)
    - Precision : TP / (TP+FP)
    - Recall : TP / (TP+FN)
    - F1 score : 2 * (Precision * Recall) / (Precision + Recall)
    - Under these evaluation metrics, we tried to enhance the scores and the performances.

___
### *Results*

- We used 836 of ***Audionautix*** source music either with 46-tags dataset set by the actual composer or with 30-tags dataset that we manually refined.
- To increase the size of the dataset, ***fma*** audio data was added to the *audionautix* source, that we were able to form a dataset based on a total of 1,136 audio, which was based on 46 different tags.
- In the final test, 836 audio sources containing only the refined 30 tags were used, and the music were distributed at a ratio of **train : valid : test = 636 : 100 : 100**.
- The followings are links for Colab with a fixed version of the dataset and a result cell executed on each model.
    - CRNN
        - w/29s: [https://colab.research.google.com/drive/15Y13LiGcvkmJxo_lwPOWKWigmwhUUmMe?usp=sharing](https://colab.research.google.com/drive/15Y13LiGcvkmJxo_lwPOWKWigmwhUUmMe?usp=sharing)
        - w/3s: [https://colab.research.google.com/drive/1oIfgXjR7cje3QylbZ1BO790fGLYsqD9N?usp=sharing](https://colab.research.google.com/drive/1oIfgXjR7cje3QylbZ1BO790fGLYsqD9N?usp=sharing)
    - Musicnn
        - w/29s: [https://colab.research.google.com/drive/13RB4MtXBGyDXpnv34bIO3W34plni0hYt?usp=sharing](https://colab.research.google.com/drive/13RB4MtXBGyDXpnv34bIO3W34plni0hYt?usp=sharing)
        - w/3s: [https://colab.research.google.com/drive/1iHBg_qxZs4qstUxeLTE0AbFavmcjraXq?usp=sharing](https://colab.research.google.com/drive/1iHBg_qxZs4qstUxeLTE0AbFavmcjraXq?usp=sharing)
- Scores
    
    |  | Loss | Accuracy | Precision | Recall | F1 |
    | --- | --- | --- | --- | --- | --- |
    | CRNN_29(validation) | 0.9078 | 0.8273 | 0.5440 | 0.5589 | 0.5473 |
    | CRNN_29(test) | 0.9079 | 0.8230 | 0.5340 | 0.5497 | 0.5376 |
    | CRNN_3(validation) | 0.9111 | 0.6840 | 0.4806 | 0.4600 | 0.4487 |
    | CRNN_3(test) | 0.9134 | 0.6587 | 0.4793 | 0.4432 | 0.4392 |
    | MusiCNN_29(validation) | 0.8958 | 0.7837 | 0.5443 | 0.5767 | 0.5368 |
    | MusiCNN_29(test) | 0.8998 | 0.7683 | 0.5273 | 0.5607 | 0.5200 |
    | MusiCNN_3(validation) | 0.9081 | 0.7743 | 0.5447 | 0.5799 | 0.5418 |
    | MusiCNN_3(test) | 0.9089 | 0.7770 | 0.5545 | 0.5994 | 0.5523 |

- Prediction
    - We predicted the tags for the song named *Sunday Spirit* from *Audionautix* using each model, and the results are as follows. You can listen to the mood of *Sunday Spirit* through the link below.
    - [https://drive.google.com/file/d/1k9aFCZ5FRpKj2WgcZrND6B3Vr6RvNvv_/view?usp=sharing](https://drive.google.com/file/d/1k9aFCZ5FRpKj2WgcZrND6B3Vr6RvNvv_/view?usp=sharing)

✔️ CRNN_29

![crnn_29](https://user-images.githubusercontent.com/76762181/173187211-5b47980f-6ec4-4000-a9e5-44f07253548a.png)

✔️ CRNN_3

![crnn_3](https://user-images.githubusercontent.com/76762181/173187209-48450554-992a-461a-921a-f16bdbe01f7c.png)

✔️ MusiCNN_29

![musicnn_29](https://user-images.githubusercontent.com/76762181/173187215-17f203e8-fb73-4e21-ab1e-5b7efa4ff95f.png)

✔️ MusiCNN_3

![musicnn_3](https://user-images.githubusercontent.com/76762181/173187214-a4e0fc35-377a-4742-af99-eb3ec778e790.png)


___
### *Analysis*
- Comparing CRNN models trained on data of different lengths, CRNN_29 had much better accuracy and f1 score on test dataset than CRNN_3. They have the values of 0.8230 and 0.6587 for accuracy, and 0.5376 and 0.4392 for f1 score, respectively. We have predicted that it is somewhat natural that longer piece of music represent the mood of the whole music better than shorter pieces of music. Since there are some cases that a single piece of music can have different moods (e.g., music that starts with a quiet intro and has a grand climax) or that multiple pieces of music make similar mood, a model trained with longer audio data would perform better. Thus, we can infer that the length of music used for training affects the performance of the model.

- Comparing MusiCNN models trained on data of different lengths, on the other hand, showed opposite results to CRNN. MusiCNN_29 had slightly smaller accuracy and quite smaller f1 score on test dataset than MusiCNN. They have the values of 0.7683 and 0.7770 for accuracy, and 0.5200 and 0.5523 for f1 score, respectively. We thought that a long dataset increases the compression rate in CNN, preventing it from storing much information about music. Since, the difference is not significant, we whould say they have similar and consistent training accuracy. We analyzed that the musically motivated CNN and the layers of MusiCNN captures feature well and consistently shows around 77% accuracy performance. That is, although we could have modified the layers even more to enhance the performance, the current layers are capturing features well, regardless of the length of the audio piece.

- Comparing CRNN model and MusiCNN model was also worthwhile. When it comes to traning with 29s-dataset, CRNN showed better performance, while when it comes to 3s-dataset, MusiCNN showed better performance. This indicates that with shorter audio data chunks, MusiCNN predicts the mood of the song better than CRNN. Also, we can find out that when the audio data chunks get long enough, CRNN will perform better than MusiCNN. Then, comparing CRNN_29 and MusiCNN_3 comes next. Although CRNN_29 had higher accuracy, MusiCNN_3 had higher values for the rest, including f1 score. So we can conclude that MusiCNN_3 has better performance than CRNN_29.


___
### *Reference*

- Data Set
    - [https://paperswithcode.com/dataset/magnatagatune](https://paperswithcode.com/dataset/magnatagatune)
    - [https://audionautix.com/](https://audionautix.com/)
    - [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
- Data Pre-processing
    - [https://dataunbox.com/split-audio-files-using-python/](https://dataunbox.com/split-audio-files-using-python/)
- Prior Research
    - [https://www.researchgate.net/publication/220723625_Tag_Integrated_Multi-Label_Music_Style_Classification_with_Hypergraph](https://www.researchgate.net/publication/220723625_Tag_Integrated_Multi-Label_Music_Style_Classification_with_Hypergraph)
    - [https://github.com/minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models)
