import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import IPython.display as ipd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Loading the audio files and their labels
train_audio_path = '../data/speech_commands_v0.01'

# Labels
labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Preprocessing the audio waves
# 1. Resampling
# 2. Removing shorter commands of less than 1 second
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        
        samples = librosa.resample(samples, orig_sr = sample_rate, target_sr = 8000)

        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)


print(type(all_wave))
print(len(all_wave))

print(len(all_wave))
print(all_wave[1].shape)
all_wave[1]


# Convert the output labels to integer encoded
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

# Convert integer encoded labels to one-hot vectors
y=np_utils.to_categorical(y, num_classes=len(labels))


# Reshape the 2D array to 3D since the input to the conv1d must be a 3D array
all_wave = np.array(all_wave).reshape(-1,8000,1)
print(all_wave.shape)

# Split into training and validation set
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y), stratify=y, test_size = 0.2, random_state=777, shuffle=True)

# %%
# Model Architecture
inputs = Input(shape=(8000,1))

conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Flatten()(conv)

conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()


# Compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# Training the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

# save model
model.save("SpeechRecogModel.h5")

# load model
# model = load_model('SpeechRecogModel.h5')


# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# Making predictions on validation data
def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

import random
index=random.randint(0, len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
print("Text:",predict(samples))

ipd.Audio(samples, rate=8000)


# Results visualization
# Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_val)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)

# Plotting the confusion matrix
cm = pd.DataFrame(cm , index = labels , columns = labels)
plt.figure(figsize = (8,6))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
plt.xlabel("Predicted")
plt.ylabel("Actual")


#Accuracy, Precision, Recall, F1-Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred , average = 'weighted')
recall = recall_score(y_true, y_pred , average = 'weighted')
f1 = f1_score(y_true, y_pred , average = 'weighted')

print("Accuracy : ", accuracy)
print("Precision : ", precision)
print("Recall : ", recall)
print("F1-Score : ", f1)
