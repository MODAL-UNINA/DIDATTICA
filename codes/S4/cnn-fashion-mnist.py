import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# visualize the first 5 images, with their classes and labels
fashion_mnist_labels = ["T-shirt/top",  # label 0
                        "Trouser",      # label 1
                        "Pullover",     # label 2
                        "Dress",        # label 3
                        "Coat",         # label 4
                        "Sandal",       # label 5
                        "Shirt",        # label 6
                        "Sneaker",      # label 7
                        "Bag",          # label 8
                        "Ankle boot"]   # label 9


# plot
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(fashion_mnist_labels[y_train[i]])
    plt.axis('off')
plt.show()


# scale the data to [0, 1] by dividing by 255.0
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# split into train and validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]


x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_val = x_val.reshape((x_val.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

print('x_train shape:', x_train.shape)


# convert y values to one-hot
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y_train[0])


# build the model
model = models.Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()

# compile the model
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])


# train the model
history = model.fit(x_train, y_train, epochs=30, batch_size=256,
                    validation_data=(x_val, y_val))


# plot the training and validation loss
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss=history.history['val_loss']

epochs = range(1, len(acc) + 1)

# plot accuracy curves.
plt.figure(figsize=(6, 4))
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend()

# plot loss curves.
plt.figure(figsize=(6, 4))
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.show()


# evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: %.4f' % (test_acc))
