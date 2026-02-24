#réseau

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_q_network() -> keras.Model:
    inp = keras.Input(shape=(6, 7, 2), dtype=tf.float32)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(7, activation=None)(x)  # Q-values pour 7 colonnes

    model = keras.Model(inp, out)
    return model
