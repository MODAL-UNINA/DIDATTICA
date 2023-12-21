import tensorflow as tf
import matplotlib.pyplot as plt

# load MNIST dataset.
# MNIST includes 60,000 training examples and 10,000 test examples.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert the samples from integers to floats.
# scale the data to the range [0.0, 1.0].
x_train, x_test = x_train/255.0, x_test/255.0

# split dataset into training and validation.
# 50000 training examples and 10000 validation examples.
x_train, x_val = x_train[:50000], x_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]


# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dense(units=10)
])


# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])


# training.
history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))


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
plt.legend()

# plot loss curves.
plt.figure(figsize=(6, 4))
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend()
plt.show()


# evaluate the model
model.evaluate(x_test, y_test, verbose=2)
