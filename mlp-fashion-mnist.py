import tensorflow as tf
import matplotlib.pyplot as plt

# load Fashion-MNIST dataset.
# Fashion-MNIST includes 60,000 training examples and 10,000 test examples.
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# convert the samples from integers to floats.
# scale the data to the range [0.0, 1.0].
x_train, x_test = x_train/255.0, x_test/255.0

# split dataset into training and validation.
# 50000 training examples and 10000 validation examples.
x_train, x_val = x_train[:50000], x_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]


# hyperparameter
LEARNING_RATE = 0.01
EPOCHS = 20
BATCH_SIZE = 256
NUM_UNITS_1 = 32
NUM_UNITS_2 = 10
Drop_rate = 0.2
ACTIVATION = 'relu'


# build model.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(NUM_UNITS_1, activation=ACTIVATION),
  tf.keras.layers.Dropout(rate=Drop_rate),
  tf.keras.layers.Dense(NUM_UNITS_2)])


# define loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss_fn, metrics=['accuracy'])


# training.
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))


# plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(acc, label='Training accuracy', color='tab:blue')
ax[0].plot(val_acc, label='Validation accuracy', color='tab:orange')
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Accuracy', fontsize=14)
ax[0].legend()
ax[1].plot(loss, label='Training loss', color='tab:blue')
ax[1].plot(val_loss, label='Validation loss', color='tab:orange')
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Loss', fontsize=14)
ax[1].legend()
plt.show()


from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# evaluate the model.
y_pred = model.predict(x_val)
y_pred = np.argmax(y_pred, axis=1)

# precision, recall, f1-score.
precision, recall, f1_score, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1-score: {:.2f}'.format(f1_score))

# confusion matrix.
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 4), dpi=150)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.show()
