#!/usr/bin/env python
# coding: utf-8

# # LSTM model for emotion detection on CMU Panoptic dataset
# The LSTM model has 30 states in this script, corresponding to the 30 frames within one second.
# 
# ## Input
# trainX : 3-d array with shape: (# of seconds in total, # of frames/states, dimension of features), in office1 should be (5346,30,76)
# 
# trainY : 1-d array with shape: (# of seconds in total,1)
# 
# ## Output
# 
# xxx
# 
# ## Multipule persons
# Notice, the LSTM model here is only for single person. 
# The task for multiple person detection is conducted by openpose.
# So, in the training stage, even if there are multiple persons in one sequence, we split them apart and train the LSTM respectively. 
# Different persons' skeletons are a kind of data augmentation here.
# While in the demo stage, we will assign each person emerging in the camera a LSTM. 
# 
# ## Questions left
# Treat different cameras as different epochs?

# In[1]:


import json
import numpy as np
import pandas as pd
import os
# from IPython.core.debugger import set_trace
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from time import time


# In[2]:


# Setup the paths
data_path = '../'
seq_name = '170915_office1'#5376
# seq_name = '170407_office2' #3649
camera_name='0' #could be 0-30
hd_skel_json_path=data_path+seq_name+"/camerawise_skeleton/"
fps=30


# In[3]:


# Prepare the trainX and trainY 
'''
 Load the dataset and labels into the X and Y with the form as described above
 The calibrated skeletons for each person are stored in :
 ../170915_office1/camerawise_skeleton/hd_0_0(0-30)_samples_for_persons.json
 '''

# do one camera as the start
# with open(hd_skel_json_path+"hd_0_"+camera_name+"_samples_for_persons.json") as skeleton_json:
#     skeletons=json.load(skeleton_json)

cameras_skeletons_list=list()

json_files = [pos_json for pos_json in os.listdir(hd_skel_json_path) if pos_json.endswith('.json')]
for index, js in enumerate(json_files):
    try:
        # Load the json file with this frame's skeletons
        with open(os.path.join(hd_skel_json_path, js)) as json_file:
            skeletons = json.load(json_file)
        cameras_skeletons_list.append(skeletons)
    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror) 

# load the labels
df = pd.read_excel (data_path+seq_name+'/office1_label.xlsx')
labels=df.as_matrix()# (68,2) ndarray
seconds=list(labels[:,0])[0:63]# extract all the useful seconds
emotion_label=labels[:,1][0:63]# extract all the useful labels

# train_seconds=seconds[0:40]
# train_emotion_label=emotion_label[0:40]

# test_seconds=seconds[40:]
# test_emotion_label=emotion_label[40:]


# convert the seconds into the frame indexes, the transform equation is:
# frames=30*seconds + frame
frames_indices=list()
for second in seconds:
    for inner_second_frame in range(fps):
        frames_indices.append(fps*second+inner_second_frame)



# filter the seconds that only appear in the labels
# convert the skeletons list into ndarray 
person_trainX=dict()
for skeletons in cameras_skeletons_list:
    for person, skels in skeletons.items():
        filtered_skels=list()
        for i in range(len(skels)):
            if i in frames_indices:
                filtered_skels.append(skels[i])
        
        if person not in person_trainX.keys():
            person_trainX[person]=list()
            
            person_trainX[person].extend(filtered_skels)#######
        else:
            person_trainX[person].extend(filtered_skels)
        
for person, samples in person_trainX.items():
    person_trainX[person]=np.array(person_trainX[person]).reshape((-1,fps,76))# person_trainX[person] store the trainX

# set_trace()
    
# One person for trainng
y=list()
[y.extend(emotion_label) for i in range(31)]
# y=to_categorical(np.array(y))
y=np.array(y)
X=person_trainX['0']
# trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)

# trainX=person_trainX['0'][0:40]
# trainy=to_categorical(np.array(emotion_label)[0:40])

# testX=person_trainX['0'][40:]
# testy=to_categorical(np.array(emotion_label)[40:])


# In[ ]:


# define 10-fold cross validation test harness
seed=7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
cvtime=[]
verbose, epochs, batch_size = 0, 60, 10
for train, test in kfold.split(X, y):
    
    # LSTM model
    
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[0]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    t0 = time()
    model.fit(X[train], to_categorical(y[train]), epochs=epochs, batch_size=batch_size, verbose=verbose)
    t1 = time()
    scores=model.evaluate(X[test], to_categorical(y[test]), batch_size=batch_size, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    cvtime.append(t1-t0)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
print("average time spent on fitting for each cv with epoch=%d, batchsize=%d is: %8f" %(epochs,batch_size,numpy.mean(time)))
# print(cross_val_score(model, X, y,scoring='recall', cv=10,fit_params={'epochs':epochs,'batch_size':batch_size,'verbose':verbose}))  
# model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

# evaluate model
# _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# t2 = time()
# print ('10 fold CV takes %f secs' %(t1-t0))
# print ('Model evaluating takes %f secs' %(t2-t1))
# print('The accuracy is: %8f' %accuracy)






