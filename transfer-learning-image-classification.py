import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# define image size, batch size, epochs
img_width, img_height = 256, 256
batch_size = 128
epochs = 30

# path to the dataset
DATASET_PATH = '../data/Food-5K/'

training_dir = DATASET_PATH + 'training'
valid_dir = DATASET_PATH + 'validation'
test_dir = DATASET_PATH + 'evaluation'

# data generators
# horizontal_flip is for randomly flipping half of the images horizontally.

# load train set
train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(
                        training_dir,
                        target_size = (img_height, img_width),
                        batch_size = batch_size, 
                        class_mode = 'binary')

# load validation set
valid_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True)

valid_generator = valid_datagen.flow_from_directory(
                        valid_dir,
                        target_size = (img_height, img_width),
                        batch_size = batch_size, 
                        class_mode = 'binary')

# load test set
test_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True)  

test_generator = valid_datagen.flow_from_directory(
                        test_dir,
                        target_size = (img_height, img_width),
                        batch_size = 16, 
                        class_mode = 'binary')


# output image samples, to check if the data is loaded correctly.
imgs, labels = next(test_generator)
num_figs = 16

fig, axs = plt.subplots(2, 8, figsize=(20, 6))
for i in range(num_figs):
    ax = axs[i//8, i%8]
    ax.imshow(imgs[i])
    ax.set_title(f'class: {int(labels[i])}', fontsize=14)
    ax.axis('off')


# load VGG model.
# include_top=False means that the fully connected layers are not loaded.
model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

model.summary()

# freeze layers of pre-trained model
for layer in model.layers:
    layer.trainable = False


# add custom Layers
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

# creating the final model
model_final = Model(model.input, predictions)

# compile the model
model_final.compile(loss='binary_crossentropy',
                optimizer = 'adam',
                metrics=['accuracy'])

model_final.summary()

# save the best model according h5 file. 
checkpoint_file = "vgg16_food5k.h5"

checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', period=1)

# train the model 
history = model_final.fit_generator(train_generator,
                        steps_per_epoch = train_generator.n // batch_size,
                        epochs = epochs,
                        validation_data = valid_generator,
                        validation_steps = valid_generator.n // batch_size,
                        callbacks = [checkpoint],
                        verbose = 1)

# plot the training and validation loss + accuracy
fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].plot(history.history['loss'], label='Train', )
axs[0].plot(history.history['val_loss'], label='Valid')
axs[0].set_title('Loss', fontsize=14)
axs[0].legend()
axs[1].plot(history.history['accuracy'], label='Train')
axs[1].plot(history.history['val_accuracy'], label='Valid')
axs[1].set_title('Accuracy', fontsize=14)
axs[1].legend()
fig.show()

# evaluation using stored model
model_final.load_weights(checkpoint_file)

loss, acc = model_final.evaluate_generator(test_generator, steps = test_generator.n // test_generator.batch_size)
print(f'Test Acc: {acc}')

