import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

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
                    keras.losses.binary_crossentropy(data, reconstruction), axis=None
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


class VariationAutoEncoder1d():
    """
    Helper class function to make it easier to run variational autoencoders on 1d vectors
    Based on https://keras.io/examples/generative/vae/ but for 1D arrays and easier
    """
    _latent_dim: int
    _input_dim: int
    _encoder = None
    _decoder = None
    _model = None

    def __init__(self, latent_space_dim: int, input_vector_size: int, encoder = None, decoder = None) -> None:
        self._latent_dim = latent_space_dim
        self._input_dim = input_vector_size

        # Define or use encoder
        if not encoder:
            self.create_encoder()
        
        # Define or use decoder
        if not decoder:
            self.create_decoder()

        self.define_model()

    def create_encoder(self):
        """
            Helper function that generates simple fully conneceted encoder
        """
        # This is our input image
        input_img = keras.Input(shape=(self._input_dim,), name="Original Input")

        # Fully connected to a mean and variance
        z_mean = layers.Dense(self._latent_dim, name="z_mean")(input_img)
        z_log_var = layers.Dense(self._latent_dim, name="z_log_var")(input_img)

        # Custom Sampleling
        z = Sampling()([z_mean, z_log_var])
        self._encoder = keras.Model(input_img, [z_mean, z_log_var, z], name="encoder")

    def create_decoder(self):
        # Helper function to generate simple connected decoder
        latent_inputs = keras.Input(shape=(self._latent_dim,))
        decoded = layers.Dense(self._input_dim, activation='sigmoid', name="Decoded")(latent_inputs)
        self._decoder = keras.Model(latent_inputs, decoded, name="decoder")
        
    def define_model(self, optimizer = keras.optimizers.Adam()):
        """
            Defines the VAE model
        """
        if not self._encoder or not self._decoder:
            raise ValueError("Encoder or Decoder are undefined")

        self._model = VAE(self._encoder, self._decoder)
        self._model.compile(optimizer=optimizer)

    def fit(self, train_data: np.array, epochs: int=10, batch_size:int = 128):
        self._model.fit(train_data, epochs=epochs, batch_size=batch_size)

    def encode(self, input_array:np.array) -> np.array:
        """
            Helper function that encodes the input array, which should be the unmodified arary
        """
        if len(input_array.shape) == 1:
            input_array = input_array.reshape((1,-1))

        return self._encoder.predict(input_array)
    
    def decode(self, input_array:np.array) -> np.array:
        """
            Helper function that decodes the input array, which is the latent space array
        """
        return self._encoder.predict(input_array)

    def plot_encoded_space(self, input_array:np.array, ax = None, x_axis_idx: int = 0, y_axis_idx:int = 1):
        """
            Takes as input an unmodified input array and plots its encoded space onto a 2d plot
        """
        latent_array = self.encode(input_array)

        if not ax:
            fig, ax = plt.subplots(1,1)

        ax.plot(latent_array[0][:,x_axis_idx], latent_array[0][:,y_axis_idx], '.') 
        return ax

    def plot_example(self, input_array:np.array, idx:int=None, ax = None):
        if not idx:
            idx = random.randint(0, len(input_array))

        if not ax:
            fig, ax = plt.subplots(1,1)

        sub_array = input_array[idx, ]
        latent_array = self.encode(sub_array)
        ax.plot(sub_array, '.')
        ax.set_title('{}: ({:1.1f}, {:1.1f})'.format(idx, latent_array[0][0][0], latent_array[0][0][1]))


        


        
