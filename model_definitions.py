import tensorflow as tf
from tensorflow.keras import layers, Model

class SimpleAutoencoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(30, activation="relu"),
                layers.Dense(15, activation="relu"),
                layers.Dense(7, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(7, activation="relu"),
                layers.Dense(15, activation="relu"),
                layers.Dense(30),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class DeepAutoencoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(30, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(20, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Dense(5, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(5, activation="relu"),
                layers.Dense(10, activation="relu"),
                layers.Dense(20, activation="relu"),
                layers.Dense(30),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class SparseAutoencoder(Model):
    def __init__(self, l1_lambda=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda

        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(30, activation="relu"),
                layers.Dense(15, activation="relu"),
                layers.Dense(7, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(7, activation="relu"),
                layers.Dense(15, activation="relu"),
                layers.Dense(30),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        self.add_loss(self.l1_lambda * tf.reduce_sum(tf.abs(encoded)))
        return decoded
