# steps required 
# 1. clean data - removing images which not appear alike to emotion 
# 2. prepare data - make the number of images equal in every category 
# 3. train model
# 4. test model 

import os 
from util import get_face_landmarks
import cv2
import numpy as np

output=[]
data_dir=r'D:\computer vision\emotion_detection\emotion_detection_dataset\train'

for emotion_indx,emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir,emotion)):
        image_path=os.path.join(data_dir,emotion,image_path_)
        
        image=cv2.imread(image_path)
        
        face_landmarks=get_face_landmarks(image)
        
        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)
            
np.savetxt('data2.txt', np.asarray(output))