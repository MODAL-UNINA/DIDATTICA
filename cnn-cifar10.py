import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.layers import MaxPooling2D


# load dataset
cifar = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar.load_data()

# normalize pixel values to be between 0 and 1
X_train = X_train/255.0
X_test = X_test/255.0


# split dataset into 90% training examples and 10% validation examples.
X_train, X_val = X_train[:45000], X_train[45000:]
y_train, y_val = y_train[:45000], y_train[45000:]


# one hot encode target values
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)


# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding = "same", input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding = "same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding = "same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# summarize model, to see the architecture of the model
model.summary()


# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# define early stopping callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# fit model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=256, 
                    validation_data=(X_val, y_val), 
                    callbacks=[callback])


# plot learning curves
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(5, 3))
plt.title('Loss')
plt.plot(epochs, history.history['loss'], label='train')
plt.plot(epochs, history.history['val_loss'], label='validation')
plt.legend()

plt.figure(figsize=(5, 3))
plt.title('Accuracy')
plt.plot(epochs, history.history['accuracy'], label='train')
plt.plot(epochs, history.history['val_accuracy'], label='validation')
plt.legend()


# evaluation on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
