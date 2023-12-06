import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

# load dataset IMDB 
imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words=10000 means only keep the top 10000 most frequently occurring words in the training data.

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])


# Transform the integers back to words
# A dictionary mapping words to integer indices.
word_index = imdb.get_word_index()

# keep the first index
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Display the text of the first review by calling the decode_review function like that:
print(decode_review(train_data[0]))


train_data = sequence.pad_sequences(train_data,
                                    value=word_index["<PAD>"],
                                    padding='post',
                                    maxlen=256)

test_data = sequence.pad_sequences(test_data,
                                   value=word_index["<PAD>"],
                                   padding='post',
                                   maxlen=256)

# check the shape like that:
print(len(train_data[0]))
print(train_data[0])


# build the model
vocab_size = 10000

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# evaluate the model
results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)

