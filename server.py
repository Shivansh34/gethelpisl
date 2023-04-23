import pickle
import cv2
from flask import Flask, request
from PIL import Image
import numpy as np
import tensorflow as tf
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Now second step is to set the hands function which will hold the landmarks points
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


model = pickle.load(open('modelconv_95.h5','rb'))

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


res = [.2,0.7,.1,.1,.1,.1,.1]
#New Detection Variables
sequence = []
sentence = []
threshold = .4
# Load the pre-trained ML classification model
# model = tf.keras.models.load_model('path/to/your/model')

@app.route('/',methods=['GET'])
def default():
    return {"res":'running'}

@app.route('/predict', methods=['POST'])
def predict():

    image_file = request.files['image']

    # Load the image from the file object
    img = Image.open(image_file.stream)

    # Preprocess the image
    img = img.resize((224, 224))


    image,results = mediapipe_detection(img,hands)
    
    #Prediciton Logic


    hand_coords = [[],[]]
    cv2.waitKey(10)
    if not(results.multi_hand_landmarks):
        cv2.imshow('iamg',image)
        return {'predicted_class': 'none'}
    

    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if(hand_no>1):
            break
        hand_coords[hand_no].append(np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten())
        
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
        #     mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
    

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
    
    ans = np.argmax(res)

    return {'predicted_class': ans,"sentence":sentence}
    
    
    # #Viz probability
    # image = prob_viz(res,actions,image,colors)
    
        
    # cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)
    # cv2.putText(image, ' '.join(sentence),(3,30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    
    # Receive the image data as input
    # image_file = request.files['image']

    # # Load the image from the file object
    # img = Image.open(image_file.stream)

    # # Preprocess the image
    # img = img.resize((224, 224))
    # img = np.array(img) / 255.0
    # img = np.expand_dims(img, axis=0)

    # Make predictions using the ML model
    # predictions = model.predict(img)

    # # Get the class label with the highest probability
    # predicted_class = np.argmax(predictions[0])

    # # Return the predictions as a JSON response
    # return {'predicted_class': predicted_class}

if __name__ == '__main__':
    app.run(debug=True)
