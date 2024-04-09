import numpy as np
import pandas as pd
import keras

# time, 16 joint angles, 8 channels
df = pd.read_csv("data.csv")

x_train = df.iloc[:, 1:17]
y_train = df.iloc[:, 17:]


def identity_block(x):
    x_skip = x

    x = keras.layers.Conv2D(1, kernel_size=(3, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.05)(x)
    x = keras.layers.Add()([x, x_skip])

    return x


def neuropose():
    x_input = keras.layers.Input(shape=(1000, 8, 1))
    x = x_input

    # Encoder
    filter_count = [32, 128, 256]
    pool_sizes = [(5, 2), (4, 2), (2, 2)]

    for filter_size, pool_size in zip(filter_count, pool_sizes):
        x = keras.layers.Conv2D(filter_size, kernel_size=(3, 2), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(0.05)(x)
        x = keras.layers.MaxPooling2D(pool_size=pool_size)(x)

    # ResNet
    x = identity_block(x)
    x = identity_block(x)
    x = identity_block(x)

    # Decoder
    filter_count = [256, 128, 32]
    pool_sizes = [(5, 4), (4, 2), (2, 2)]

    for filter_size, pool_size in zip(filter_count, pool_sizes):
        x = keras.layers.Conv2D(filter_size, kernel_size=(3, 2), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(0.05)(x)
        x = keras.layers.UpSampling2D(size=pool_size)(x)

    x = keras.layers.Flatten()(x)
    x_output = keras.layers.Dense(16)(x)

    model = keras.models.Model(inputs=x_input, outputs=x_output, name="NeuroPose")

    return model


model = neuropose()

model.summary()
