import shutil
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
# Create a hand landmarker instance with the image mode:

mp_hands = mp.solutions.hands

# mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2)
                             )
    
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                             )

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh,rh])


DATA_PATH = os.path.join('processed_data')
#Actions
actions = np.array(['accident','call','doctor','help','hot','lose','pain','thief'])


# create normalized data

local_data_path = os.path.join('data\Raw_Data')
normalized_data_path = os.path.join('normalized')

arr = []

normalized_frames = 60
 
def normalize_data():
    for action in os.listdir(local_data_path):
        action_path = os.path.join(local_data_path,action)
        os.mkdir(os.path.join(normalized_data_path,action))
        for video_name in os.listdir(action_path):
            video_path = os.path.join(action_path,video_name)
            # print(video_path)
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            path= os.path.join(normalized_data_path,action,video_name)
            out = cv2.VideoWriter(path,fourcc,fps,(frame_width,frame_height))
            print(path)
            if total_frames < normalized_frames:
                extra_frames_required = normalized_frames-total_frames
                for _ in range(total_frames):
                    ret,frame = cap.read()
                    if ret:
                        out.write(frame)
                        if extra_frames_required>0:
                            out.write(frame)
                            # cv2.imshow('frams',frame)
                            # if cv2.waitKey(30) & 0xFF == ord('q'):
                            #     break
                            extra_frames_required -= 1
            else:
                for _ in range(normalized_frames):
                    ret,frame = cap.read()
                    if ret:
                        out.write(frame)
            cap.release()
            out.release()
    
# normalize_data()


# Create a hand landmarker instance with the image mode:
def create_processed_data_with_hand_soluntion():
    mp_hands = mp.solutions.hands

    # Now second step is to set the hands function which will hold the landmarks points
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3,min_tracking_confidence=0.3)

    # Last step is to set up the drawing function of hands landmarks on the image
    mp_drawing = mp.solutions.drawing_utils

    # shutil.rmtree(DATA_PATH)
    os.mkdir(DATA_PATH)
    for action in os.listdir(normalized_data_path):
        action_path = os.path.join(normalized_data_path,action)
        os.mkdir(os.path.join(DATA_PATH,action))
        print(action)
        for video_name in os.listdir(action_path):
            print(video_name)
            video_path = os.path.join(action_path,video_name)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # make directort
            new_path = os.path.join(DATA_PATH,action,os.path.splitext(video_name)[0])
            os.mkdir(new_path)
            # print(total_frames)
            
            prev_hands_present=[0,0]
            prev_hands=[[],[]]
            prev_frame=[0,0]

            for frame_num in range(total_frames):
                ret, frame = cap.read()
                image,results = mediapipe_detection(frame,hands)
                
                hand_coords = [[],[]] 

                hands_present=[0,0]
                if not(results.multi_hand_landmarks):
                    break

                for a in results.multi_handedness:
                    if(a.classification[0].label=='Left' and a.classification[0].score>0.85):
                        # print(a.classification)
                        prev_hands_present[0]=1
                        hands_present[0]=1
                    else:
                        # print(a.classification)
                        prev_hands_present[1]=1
                        hands_present[1]=1
                
                # print(hands_present)
                
                hands_recorded=[0,0]
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if(hand_no>1):
                        break

                    if(hands_present[0] and hands_present[1]):
                        hand_no=hand_no
                    elif(hands_present[0]):
                        hand_no=0
                    else:
                        hand_no=1
                    
                    if(hands_recorded[hand_no]):
                        continue
                    else:
                        hands_recorded[hand_no]=1
                    
                    prev_hands[hand_no]=hand_landmarks
                    prev_frame[hand_no]=frame_num

                    hand_coords[hand_no].append(np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten())
                
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
                    
                # print(len(hand_coords[0]),len(hand_coords[1]))
                # print(len(hand_coords[0][0]) if len(hand_coords[0])!=0 else "," ,len(hand_coords[1][0]) if len(hand_coords[1])!=0 else ",")
                # print(hands_present,prev_hands_present)

                for hand_no in range(len(hands_present)):
                    if(not(prev_hands_present[hand_no]) or hands_present[hand_no] or abs(prev_frame[hand_no]-frame_num)>3):
                        continue
                    # else:
                    #     print(hand_no,frame_num,video_name)

                    hand_landmarks=prev_hands[hand_no]
                    if(type(hand_landmarks)==type([])):
                        continue
                    # else:
                    #     print(hand_no,frame_num,video_name)
                    #     print(hand_landmarks.landmark)

                    hand_coords[hand_no].append(np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten())
                
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1))
                
                # print(len(hand_coords[0][0]) if len(hand_coords[0])!=0 else "," ,len(hand_coords[1][0]) if len(hand_coords[1])!=0 else ",")

                if len(hand_coords[0])==0:
                    hand_coords[0].append(np.array(np.zeros(21*3)))
                
                if len(hand_coords[1])==0:
                    hand_coords[1].append(np.array(np.zeros(21*3)))

                keypoints = np.concatenate([hand_coords[0][0],hand_coords[1][0]])
                # print(len(keypoints),"/n")
                npy_path = os.path.join(new_path,str(frame_num))
                # print(npy_path)
                np.save(npy_path,keypoints)
                # print(keypoints)
                
                # cv2.imshow('OpenCV feed', frame)
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     exit(0)
            # break
            cap.release()

    cv2.destroyAllWindows()



create_processed_data_with_hand_soluntion()


def create_processed_data_with_holistic():
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2,static_image_mode=False,smooth_landmarks=True)
    shutil.rmtree(DATA_PATH)
    os.mkdir(DATA_PATH)

    for action in os.listdir(normalized_data_path):
        action_path = os.path.join(normalized_data_path,action)
        os.mkdir(os.path.join(DATA_PATH,action))
        print(action)
        for video_name in os.listdir(action_path):
            print(video_name)
            video_path = os.path.join(action_path,video_name)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # make directort
            new_path = os.path.join(DATA_PATH,action,os.path.splitext(video_name)[0])
            os.mkdir(new_path)

            for frame_num in range(total_frames):
                ret, frame = cap.read()
                image,results = mediapipe_detection(frame,holistic)

                # draw_styled_landmarks(image,results)
                
                # #Wait Logic
                # if frame_num==0:
                #     cv2.putText(image,'Starting Collection',(120,200),
                #                 cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                #     cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,video_name),(15,12),
                #                 cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
                #     cv2.waitKey(2000)
                
                # else:
                #     cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action,video_name),(15,12),
                #                 cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)

                #NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(new_path,str(frame_num))
                np.save(npy_path,keypoints)
                
                #Show to Screen
                # cv2.imshow('OpenCV feed', image)

                # #Breaking the Feed
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break

    cap.release()
    cv2.destroyAllWindows() 

# create_processed_data_with_holistic()

