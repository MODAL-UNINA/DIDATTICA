import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# load dataset IMDB 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# check the dataset
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# check the first review
print(train_data[0])


# a dictionary mapping words to integer indices.
word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# display the text of the first review
print(decode_review(train_data[0]))


# pad the sequences to the same length
train_data = sequence.pad_sequences(train_data,
                                    value=word_index["<PAD>"],
                                    padding='post',
                                    maxlen=256)

test_data = sequence.pad_sequences(test_data,
                                   value=word_index["<PAD>"],
                                   padding='post',
                                   maxlen=256)

# check the shape
print(len(train_data[0]))
print(train_data[0])


# split the data into training and validation sets
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]


vocab_size = 10000
embedding_dim = 16

# build the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=256))
model.add(LSTM(units=16))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# train the model
history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=256,
                    validation_data=(x_val, y_val))


# plot the training and validation accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label="Validation acc")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# find the best training and validation accuracy
best_val_acc = np.max(val_acc)
best_train_acc = np.max(acc)

print("Best training accuracy: {:.4f}".format(best_train_acc))
print("Best validation accuracy: {:.4f}".format(best_val_acc))

# evaluate the model
results = model.evaluate(test_data, test_labels)

print('Test Accuracy: {:.4f}'.format(results[1]))

