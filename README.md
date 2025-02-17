# Overlapped-Speech-Detection
This is machine learning project including deep learning for overlap  speech detection

# Merging Procedure

started the learning with basic knowledge of feature extraction of audio samples using librosa library
and generation of dataset for overlapped and non overlapped using two audio files 

file : merging.ipynb
using this file you can generate the number of samples of overlapped and non overlapped audios.
here we keep the follwing length of generated samples  

min_duration = 2  # Minimum duration in seconds
max_duration = 10 # Maximum duration in seconds  
max duration depends upon the length of smallest file in terms of time,

How It Works :
Audio 1 : Say X (10 sec length)
Audio 2 : Say Y (20 sec length)

randonly select starting point of both the audios :
for generating overlap samples save audio1 + audio 2
for non overlap save concatenate(audio 1, audio 2)



# Dataset Creation Using Stereo Audio File
here take two stereo audio files and find their dominant channels, and calculate the speech and non speech intervals of the dominant channels of both the files
as the code will save information in a txt file , using those txt files data set will be generated 
Links of Files Used : 
[To be Updated ]
file : streo.ipynb

# Data Set  Generation Using Files From Grid Corpus 
folders = ["s2", "s3", "s12", "s13", "s28", "s29", "s30"]
using these folder dataset is generated for overlapped and non overlapped samples , and speaker information is saved in the txt file , so that it can be used in future 
link to grid corpus : https://zenodo.org/records/3625687
file : data_generation.ipynb

# Creating GMM Model using MFCC Features and also using Bays Classifier 

File : gmm.ipynb

# Deep Learning 

Completed basic knowledge of deep learnig its flow , 
then studied about CNN classfier ( how it works with and without hidden layer )
then implemented the CNN logic to mnist dataset to understand its functionality and libraries like tensorflow

file : mnist_cnn.ipynb

