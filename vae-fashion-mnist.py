import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Layer, Input, Reshape, Conv2DTranspose
from tensorflow.keras import optimizers, metrics
from tensorflow.keras import losses
from tensorflow.keras.models import Model


# Prepare MNIST data
(x_train, y_train), (x_test, _) = fashion_mnist.load_data()

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.


# Create sampling layers
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# dimensionality of the latent space
latent_dim = 2

# Encoder
class Encoder(Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = Flatten()
        self.dense1 = Dense(16, activation="relu")
        self.z_mean = Dense(latent_dim, name="z_mean")
        self.z_log_var = Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z
    
encoder = Encoder(latent_dim=latent_dim, name="encoder")


# Decoder
class Decoder(Model):
    def __init__(self, latent_dim=2, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense1 = Dense(7 * 7 * 64, activation="relu")
        self.reshape = Reshape((7, 7, 64))
        self.conv1 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv2 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv3 = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
    
decoder = Decoder(latent_dim=latent_dim, name="decoder")


# Define the VAE as a `Model` with a custom `train_step`
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                     losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# Train the VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer= optimizers.Adam())
vae.fit(x_train, epochs=100, batch_size=256)


# Draw a grid of latent space sample points.
def plot_latent_space(vae, n=30, figsize=6):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(12, 12))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]", fontsize=14)
    plt.ylabel("z[1]", fontsize=14)
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space(vae)


# plot label clusters in the latent space
def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(6, 6))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]", fontsize=14)
    plt.ylabel("z[1]", fontsize=14)
    plt.show()
 
plot_label_clusters(vae, x_train, y_train)
