# EmotionGestureDetection
This repository is a part of KBTG emotion detection project, which aims at combining gesture emotion detection with facial emotion detection.

## Requirement
Python 3.7.4

tensorflow 1.14.0

keras 2.2.4

## Introduction
This section introduces the contents or functions of different folders and files.
### 170915_office1 & 17040_office2
These two folders store the data for scenarios of office1 and office2, each including 2 sub-folders and 3 files

#### camerawise_skeleton

There are 31 json files under this directory, representing the augmented skeletons from 31 different HD cameras indexed from 0 to 30.

#### hdPose3d_stage1_coco19

The frame-wise skeleton labels of CMU Panoptic dataset. The count of json files under this directory equals the total frames of the video.

#### calibration_170915_office1.json & calibration_170407_office2.json
Calibration parameters for 31 HD cameras. 

#### office1_label.xlsx & office2_label.xlsx
The second-wise emotion labels. The first column is the second, the second column is the label, where 1, 2, 3 represent happiness, unhappiness and neutralism respectively. 

#### samples_for_persons.json
The skeletons from all the 31 HD camreras, stored as python dictionary object form. Its keys are person id, and values are 2-d dimension numpy ndarray.

### python
This directory stores the executable files. 
#### Extract_skels.ipynb
Extract the skeletons for each person into the matrices and filter out the seconds(set of frames) we label.

Input: 

~/170915_office1/hdPose3d_stage1_coco19/body3DScene_*.json (3737 frames)
~/170407_office2/hdPose3d_stage1_coco19/body3DScene_*.json (5529 frames)

Output: 

samples_for_persons.json

#### LSTM_emotion_31cams.ipynb
Combine the skeletons with 31 camera calibration parameters

Input: 

samples_for_persons.json     

calibration_170915_office1.json  

calibration_170407_office2.json

Output: 

camerawise_skeleton/*.json (31 files)

#### X.npy
The direct input of LSTM model, with shape of (2387,30,76).

#### y.npy
The direct input of LSTM model, with shape of (2387,1).

#### y1_list.json

The list form of y.npy

#### LSTM_emotion.ipynb
The script of LSTM model.

#### LSTM_emotion_1000epochs.py
The exported python file of LSTM_emotion.ipynb, for the convenience of running in background. 
