import pickle
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM,Dense,Conv1D,Dropout,MaxPooling1D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

mp_hands = mp.solutions.hands

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Now second step is to set the hands function which will hold the landmarks points
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results


colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(200,100,200),(100,200,100)]
def prob_viz(res,actions,input_frame,colors):
    output_frame = input_frame.copy()
    for num,prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40),colors[num], -1)
        cv2.putText(output_frame,actions[num],(0,85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    return output_frame


#Actions
actions = np.array(['accident','call','doctor','help','hot','lose','pain','thief'])

model = pickle.load(open('modelconv_95.h5','rb'))

res = [.2,0.7,.1,.1,.1,.1,.1]
#New Detection Variables
sequence = []
sentence = []
threshold = .4

cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    #Read Feed
    ret, frame = cap.read()
    
    #Make detections
    image,results = mediapipe_detection(frame,hands)
    
    #Prediciton Logic
    hand_coords = [[],[]]
    cv2.waitKey(10)
    if not(results.multi_hand_landmarks):
        cv2.imshow('iamg',image)
        continue
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if(hand_no>1):
            break
        hand_coords[hand_no].append(np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten())
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
    

    if len(hand_coords[0])==0:
        hand_coords[0].append(np.array(np.zeros(21*3)))
    
    if len(hand_coords[1])==0:
        hand_coords[1].append(np.array(np.zeros(21*3)))

    keypoints = np.concatenate([hand_coords[0][0],hand_coords[1][0]])
    
    sequence.insert(0,keypoints)
    sequence = sequence[:60]

    if len(sequence) == 60:
        res = model.predict(np.expand_dims(sequence,axis=0))[0]
        print(res)

    #Visualization
    if res[np.argmax(res)] > threshold:
        if len(sentence) > 0:
            if actions[np.argmax(res)] != sentence[-1]:
                sentence.append(actions[np.argmax(res)])
        else:
            sentence.append(actions[np.argmax(res)])
    
    if len(sentence)>5:
        sentence = sentence[-5:]
    
    
    #Viz probability
    image = prob_viz(res,actions,image,colors)
    
        
    cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)
    cv2.putText(image, ' '.join(sentence),(3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    
    #Show to Screen
    cv2.imshow('iamg', image)
    
    #Breaking the Feed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 