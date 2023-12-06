import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt

# generate sine wave
sin_wave = np.array([math.sin(x) for x in np.arange(200)])

# plot first 50 points, check the shape of sin_wave
plt.plot(sin_wave[:50])
plt.xticks([])
plt.show()


# generate train data
train_X = []      # train data
train_Y = []      # train label

seq_len = 50
num_records = len(sin_wave) - seq_len   # 150

for i in range(num_records - 50):    # 100
    train_X.append(sin_wave[i:i+seq_len])
    train_Y.append(sin_wave[i+seq_len])
    
train_X = np.array(train_X)
train_X = np.expand_dims(train_X, axis=2)

train_Y = np.array(train_Y)
train_Y = np.expand_dims(train_Y, axis=1)

# check the shape of train_X and train_Y
print(train_X.shape, train_Y.shape)


# generate validation data
val_X = []    # validation data
val_Y = []    # validation label

for i in range(num_records - 50, num_records):
    val_X.append(sin_wave[i:i+seq_len])
    val_Y.append(sin_wave[i+seq_len])

val_X = np.array(val_X)
val_X = np.expand_dims(val_X, axis=2)

val_Y = np.array(val_Y)
val_Y = np.expand_dims(val_Y, axis=1)

# check the shape of val_X and val_Y
print(val_X.shape, val_Y.shape)


# build model
model = Sequential([
    SimpleRNN(units=64, activation='relu', input_shape=(None, 1)),
    Dense(1)
    ])


# compile model
model.compile(loss='mse', optimizer='adam')
model.summary()


# train model
history = model.fit(train_X, train_Y, epochs=30, validation_data=(val_X, val_Y))


# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.legend(['train','validation'], fontsize=14)
plt.show()


# predict
predictions = model.predict(val_X)

# plot actual vs prediction
plt.plot(val_Y)
plt.plot(predictions)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('sin wave', fontsize=14)
plt.legend(['actual','prediction'], fontsize=14)
plt.show()

