### Summary:
The Hand Gestures for Emergency scenarios with Indian Sign Language Recognition project aims to develop a system that can recognise hand gestures and translate them into ISL to help people with hearing impairments in emergency scenarios. To identify hand motions recorded by a
camera, the system is built to leverage computer vision algorithms and machine learning techniques.
As part of the project, a dataset of hand gestures and their accompanying ISL interpretations will be created. This dataset will be utilized to train the machine learning model. Additionally, the
device is built with a user-friendly interface so that those with hearing loss can readily connect with emergency personnel.
Applications for the project include use in public transportation networks, healthcare facilities, and emergency response services. During emergencies, it has the ability to dramatically enhance
communication between those who have hearing impairments and the general public, resulting in higher safety and better service accessibility. Overall, this research has the potential to
significantly improve the lives of people in India who have hearing impairments.

### ML model:
1. We used a dataset that is already collected and [published](https://www.sciencedirect.com/science/article/pii/S2352340920309100).
2. The dataset was then pre-processed to remove any noise, irrelevant features and outliners.
3. We then extracted features from the pre-processed dataset using Media Pipe.These
features included hand shape, finger position, and movement patterns.
4. The ML model is then trained using the extracted features and the corresponding Indian
Sign Language interpretations. We trained the model using CNN-LSTM layer model.
5. We then added dropout layers to avoid overfitting.
6. Finally, we deployed the optimized ML model of the Hand Gestures for Emergency
Situations with Indian Sign Language Recognition system. The model is integrated with
the user interface and real-time communication system to enable seamless
communication between individuals with hearing impairments and emergency
responders.


### App:
1. Our App captures images and sends them to the flask backend.
2. The backend extracts the hand data points data points of the image and saves them.
3. The App keeps sending images to the backend, when the features stored increase 60 we send
the combined dataset to the ml model.
4. The CNN-LSTM( ML model ), classifies the input video data points of 60 frames into one
of the eight categories, we initially trained our model on.
5. The App displays the output of the ML model to the user.
