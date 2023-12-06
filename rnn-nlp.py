import os
import numpy as np
import tensorflow as tf

# Download Shakespeare dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read and decode.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))

# print to see how many unique characters in the file
print ('{} unique characters'.format(len(vocab)))


# Vectorize the text
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])


# print first 20 characters and IDs
for char,_ in zip(char2idx, range(20)):
    print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum sentence length
seq_length = 100
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# The batch method lets us easily convert these individual characters to sequences of the desired size.
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch:
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
 
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# Shuffle the data and pack it into batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
 
# Length of the vocabulary in chars
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


# Build the model.
# Use tf.keras.layers.GRU for the RNN layer.
# Use tf.keras.layers.Dense to generate the predictions.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

model.summary()


# check the shape of the output
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# check the prediction of the first sample
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

# Use sparse_categorical_crossentropy as the loss function.
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# check the loss of the first sample
example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# Compile the model
# Use the Adam optimizer.
model.compile(optimizer='adam', loss=loss)

# Configure checkpoints
# Use a tf.keras.callbacks.ModelCheckpoint to ensure that checkpoints are saved during training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Execute the training
history = model.fit(dataset, epochs=50, callbacks=[checkpoint_callback])


# Rebuild the model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Restore the weights
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# Build the model with a batch_size=1
model.build(tf.TensorShape([1, None]))
model.summary()


# Predicting
def generate_text(model, start_string):
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  temperature = 1.0

  # Evaluation loop.
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
 
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))

# print the generated text
print(generate_text(model, start_string=u"ROMEO: "))

