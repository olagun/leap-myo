import numpy as np
import pandas as pd
import keras
import keras.backend as K
import tensorflow as tf

# 0 thumb_tm,f/e
# 1 thumb_tm,aa
# 2 thumb_mcp,f/e
# 3 thumb_mcp,aa
# 4 index_mcp,f/e
# 5 index_mcp,aa
# 6 index_pip
# 7 middle_mcp,f/e
# 8 middle_mcp,aa
# 9 middle_pip
# 10 ring_mcp,f/e
# 11 ring_mcp,aa
# 12 ring_pip
# 13 pinky_mcp,f/e
# 14 pinky_mcp,aa
# 15 pinky_pip

# time, 16 joint angles, 8 channels
df = pd.read_csv("data.csv")

joint_angles = df.iloc[:, 1:17]
channels = df.iloc[:, 17:]

x_train = []
y_train = []

# 1000 samples per training example
for i in range(0, df.shape[0] - 1000):
    # EMG data
    x_training_example = channels.iloc[i : i + 1000].fillna(0)
    x_train.append(x_training_example.values)

    # Joint angles
    y_train.append(joint_angles.iloc[i + 999])


def loss_mse(indices, y_actual, y_predicted):
    return K.mean(
        K.square(
            np.take(y_actual.numpy(), indices, axis=1)
            - np.take(y_predicted.numpy(), indices, axis=1)
        ),
        axis=-1,
    )


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

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),
    metrics=["accuracy"],
    run_eagerly=True,
)

batch_size = 128
epochs = 10

print(df)
# print(y_train)

# model.fit(
#     np.array(x_train),
#     np.array(y_train),
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_split=0.15,
# )

# model.save("model.keras")
