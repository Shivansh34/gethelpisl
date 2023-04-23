import os
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Conv1D,MaxPooling1D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam



DATA_PATH = os.path.join('processed_data4_cached_mphands')
#Actions
actions = np.array(os.listdir(DATA_PATH))
# print(actions.shape[0])

#52 videos worth of data for each category
#60 frames
sequence_length = 60
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH,action)
    print(action)
    for video_name in os.listdir(action_path):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH,action, video_name,"{}.npy".format(frame_num)))
            except:
                continue
            window.append(res)
        if len(window)!=60:
            # print(video_name)
            continue
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences,dtype='float32')
y = to_categorical(labels).astype(int)

# print(len(X[0]),len(X[0][0]))
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,shuffle=True)

# print(x_train)
# print(actions.shape[0])
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

# model = Sequential()
# model.add(LSTM(256,return_sequences=True, activation='relu', input_shape=(60,126)))
# model.add(Dropout(0.2))
# model.add(LSTM(128,return_sequences=True, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(64,return_sequences=False, activation = 'relu'))
# # model.add(LSTM(64, return_sequences = False,activation='relu'))
# model.add(Dense(64,activation='relu'))
# # model.add(Dropout(0.3))
# model.add(Dense(32,activation = 'relu'))
# model.add(Dropout(0.2))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(60, 126))) # Adjust input_shape according to your data
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(LSTM(units=128))
model.add(Dropout(0.5))
model.add(Dense(actions.shape[0],activation='softmax'))

# res = [.2,0.7,.01,.01]
# print(actions[np.argmax(res)])

optimizer = Adam(lr=0.001)

model.compile(optimizer = optimizer,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]



model.fit(x_train,y_train,epochs = 100, callbacks=[tb_callback],steps_per_epoch=60,validation_data=(x_test,y_test),shuffle=True)
model.summary()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Plot training & validation accuracy values
plt.plot(model.history['categorical_accuracy'])
plt.plot(model.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

y_pred = model.predict((x_test,y_test))
# Convert predictions to labels
y_pred_labels = np.argmax(y_pred, axis=1)
# Convert test data to labels
y_test_labels = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels,labels=['accident','call','doctor','help','hot','lose','pain','thief'])
# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# pickle.dump(model,open('model.h5','wb'))

# res = model.predict(x_test)
# print(actions[np.argmax(res[1])])
# print(actions[np.argmax(y_test[1])])
# model.save('action.h5')