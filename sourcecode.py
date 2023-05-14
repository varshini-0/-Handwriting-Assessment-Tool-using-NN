import tensorflow as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

sns.set()
df = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv").astype('float32')
df.head()
df.rename(columns={'0':'label'}, inplace=True)
X = df.drop('label', axis = 1)
y = df['label']

print(X.shape)
X.head()
from sklearn.utils import shuffle
X_Shuffled = shuffle(X)

plt.figure(figsize = (10, 10))
rows, columns = 3, 3

for i in range(9):
    plt.subplot(columns, rows, i+1)
    plt.imshow(X_Shuffled.iloc[i].values.reshape(28,28), interpolation='nearest', cmap='Greys')
    
plt.show()
alphabets="abcdefghijklmnopqrstuvwxyz"
letter_name=[]
[letter_name.append(i) for i in alphabets]
name_tag = pd.DataFrame(letter_name)

plt.figure(figsize=(15,5))
sns.distplot(y,kde=False)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
alphabets="abcdefghijklmnopqrstuvwxyz"
letter_name=[]
[letter_name.append(i) for i in alphabets]
name_tag = pd.DataFrame(letter_name)

name_tag
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)
X_Shuffled = shuffle(X)

plt.figure(figsize = (10, 10))
rows, columns = 3, 3

for i in range(9):
    plt.subplot(columns, rows, i+1)
    plt.imshow(X_Shuffled.iloc[i].values.reshape(28,28), interpolation='nearest', cmap='Greys')
    
plt.show()
from keras.utils import np_utils
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(len(y.unique()), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=16, batch_size=200, verbose=2)
model.save("htr.h5")
import pickle

Pkl_Filename = "htr.pkl"  
with open(Pkl_Filename, 'wb') as file:  
pickle.dump(model, file)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL LOSS')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
plt.imshow(X_test[[18]].reshape(28,28),cmap='Greys')
prediction = model.predict(X_test[[18]]) 
print(prediction)
score = np.argmax(prediction)
score = prediction[0,score]
score = round((score*100), 4)

score
plt.imshow(X_test[[19]].reshape(28,28),cmap='Greys')
prediction = model.predict(X_test[[19]]) 
print(prediction)
score = np.argmax(prediction)
score = prediction[0,score]
score = round((score*100), 4)

score

result(prediction)
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)
pip install imutils
import imutils
import cv2
import numpy as np
def final(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:

        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (28, 28), interpolation = cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,28,28,1)
        ypred = model.predict(thresh)

        result(ypred)

    plt.imshow(image)
def result(pred):
    alphabets="abcdefghijklmnopqrstuvwxyz"
    list1=[]
    [list1.append(i) for i in range(26)]
    list2=[]
    [list2.append(i) for i in alphabets]
    dic = dict(zip(list1, list2))
    score = np.argmax(pred)
    score = pred[0,score]
    score = round((score*100), 4)

    print("Prediction: ",dic[np.argmax(pred)], "Score: ", score)
final('../input/dummys/IMG_20220317_144817__01__01__01.jpg')

final('../input/dummys/TRAIN_00003.jpg')

final('../input/dummys/TEST_0094.jpg')
