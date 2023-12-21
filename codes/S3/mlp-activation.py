import tensorflow as tf
import matplotlib.pyplot as plt

# set random seed.
tf.random.set_seed(1234)

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


#%%
# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

# training.
history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))

acc_relu = history.history['accuracy']
val_acc_relu = history.history['val_accuracy']
loss_relu = history.history['loss']
val_loss_relu = history.history['val_loss']

# evaluate the model
_, test_acc_relu = model.evaluate(x_test, y_test, verbose=2)


#%%
# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dense(10)
])

# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

# training.
history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))

acc_tanh = history.history['accuracy']
val_acc_tanh = history.history['val_accuracy']
loss_tanh = history.history['loss']
val_loss_tanh = history.history['val_loss']

# evaluate the model
_, test_acc_tanh = model.evaluate(x_test, y_test, verbose=2)


#%%
# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='sigmoid'),
  tf.keras.layers.Dense(10)
])

# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

# training.
history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))

acc_sigmoid = history.history['accuracy']
val_acc_sigmoid = history.history['val_accuracy']
loss_sigmoid = history.history['loss']
val_loss_sigmoid = history.history['val_loss']

# evaluate the model
_, test_acc_sigmoid = model.evaluate(x_test, y_test, verbose=2)


#%%
# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='leaky_relu'),
  tf.keras.layers.Dense(10)
])

# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

# training.
history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))

acc_leaky_relu = history.history['accuracy']
val_acc_leaky_relu = history.history['val_accuracy']
loss_leaky_relu = history.history['loss']
val_loss_leaky_relu = history.history['val_loss']

# evaluate the model
_, test_acc_leaky_relu = model.evaluate(x_test, y_test, verbose=2)


#%%
# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='elu'),
  tf.keras.layers.Dense(10)
])

# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn, metrics=['accuracy'])

# training.
history = model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_val, y_val))

acc_elu = history.history['accuracy']
val_acc_elu = history.history['val_accuracy']
loss_elu = history.history['loss']
val_loss_elu = history.history['val_loss']

# evaluate the model
_, test_acc_elu = model.evaluate(x_test, y_test, verbose=2)


#%%
# plot training history.
# obtain accuracy and loss.
epochs = range(1, len(acc_relu) + 1)
# plot train accuracy curves: acc_relu, acc_tanh, acc_sigmoid.
plt.figure(figsize=(6, 4))
plt.plot(epochs, acc_relu, label='Training accuracy when activation is ReLU')
plt.plot(epochs, acc_tanh, label='Training accuracy when activation is Tanh')
plt.plot(epochs, acc_sigmoid, label='Training accuracy when activation is Sigmoid')
plt.plot(epochs, acc_leaky_relu, label='Training accuracy when activation is Leaky_ReLU')
plt.plot(epochs, acc_elu, label='Training accuracy when activation is ELU')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Train accuracy', fontsize=14)
plt.legend()


# plot validation accuracy curves: val_acc_relu, val_acc_tanh, val_acc_sigmoid.
plt.figure(figsize=(6, 4))
plt.plot(epochs, val_acc_relu, label='Validation accuracy when activation is ReLU')
plt.plot(epochs, val_acc_tanh, label='Validation accuracy when activation is tanh')
plt.plot(epochs, val_acc_sigmoid, label='Validation accuracy when activation is sigmoid')
plt.plot(epochs, val_acc_leaky_relu, label='Validation accuracy when activation is Leaky_ReLU')
plt.plot(epochs, val_acc_elu, label='Validation accuracy when activation is ELU')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation accuracy', fontsize=14)
plt.legend()


# plot train loss curves: loss_relu, loss_tanh, loss_sigmoid.
plt.figure(figsize=(6, 4))
plt.plot(epochs, loss_relu, label='Training loss when activation is ReLU')
plt.plot(epochs, loss_tanh, label='Training loss when activation is Tanh')
plt.plot(epochs, loss_sigmoid, label='Training loss when activation is Sigmoid')
plt.plot(epochs, loss_leaky_relu, label='Training loss when activation is Leaky_ReLU')
plt.plot(epochs, loss_elu, label='Training loss when activation is ELU')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Train loss', fontsize=14)
plt.legend()


# plot validation loss curves: val_loss_relu, val_loss_tanh, val_loss_sigmoid.
plt.figure(figsize=(6, 4))
plt.plot(epochs, val_loss_relu, label='Validation loss when activation is ReLU')
plt.plot(epochs, val_loss_tanh, label='Validation loss when activation is Tanh')
plt.plot(epochs, val_loss_sigmoid, label='Validation loss when activation is Sigmoid')
plt.plot(epochs, val_loss_leaky_relu, label='Validation loss when activation is Leaky_ReLU')
plt.plot(epochs, val_loss_elu, label='Validation loss when activation is ELU')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Validation loss', fontsize=14)
plt.legend()


