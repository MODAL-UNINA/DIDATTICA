import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from matplotlib import pyplot as plt
import numpy as np

# download Higgs Boson dataset.
# 11 million samples, each sample with 28 features and 1 label (0 or 1, representing 2 classes).
data_file = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

# load dataset.
FEATURES = 28
ds = tf.data.experimental.CsvDataset(data_file, [float(),]*(FEATURES+1), compression_type="GZIP")

# pack features and label.
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

packed_ds = ds.batch(10000).map(pack_row).unbatch()

# select 10000 examples for training, 1000 examples for validation.
N_TRAIN = int(1e4)
N_VALIDATION = int(1e3)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

# shuffle and batch the datasets.
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
  

# define a tiny model.
model_tiny = tf.keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

# increase the capacity of the model.
model_bigger = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# use regularization to prevent overfitting.
model_l2 = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(FEATURES,), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])


# use dropout to prevent overfitting.
model_l2_dropout = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(FEATURES,), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1)
])


# define loss function.
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# compile the model.
model = model_tiny   # change this to try different models: model_bigger, model_l2, model_l2_dropout

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# training.
history = model.fit(train_ds, 
                    epochs=1000, 
                    steps_per_epoch=STEPS_PER_EPOCH, 
                    validation_data=validate_ds, 
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=200, monitor='val_loss')], 
                    verbose=2)

# plot training history.
# obtain accuracy and loss.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss=history.history['val_loss']

epochs = range(1, len(acc) + 1)

# plot accuracy curves.
plt.figure(figsize=(6, 4))
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# plot loss curves.
plt.figure(figsize=(6, 4))
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
