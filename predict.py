from myo_api import Myo, emg_mode
import leap as lp
import matplotlib.pyplot as plt
from matplotlib import animation
import plot as leapplot
from utils import (
    get_anchor_points,
    get_joint_angles,
    get_bone_lengths,
    get_points_from_angles,
    digit_labels,
)
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import math

model = tf.keras.saving.load_model("model.h5")

sample_rate = 50

angle_labels = [
    "thumb_tm,f/e",
    "thumb_tm,aa",
    "thumb_mcp,f/e",
    "thumb_mcp,aa",
    "index_mcp,f/e",
    "index_mcp,aa",
    "index_pip",
    "middle_mcp,f/e",
    "middle_mcp,aa",
    "middle_pip",
    "ring_mcp,f/e",
    "ring_mcp,aa",
    "ring_pip",
    "pinky_mcp,f/e",
    "pinky_mcp,aa",
    "pinky_pip",
]


def get_angles_from_prediction(angles):
    joint_angles = {}

    for i in range(angles.shape[1]):
        digit, angle = angle_labels[i].split("_")

        if digit not in joint_angles:
            joint_angles[digit] = {}

        joint_angles[digit][angle] = math.radians(angles[0, i])

    return joint_angles


def leap_process_data(data, leap_data, myo_data):
    if not len(data.hands) or not len(myo_data):
        return

    hand = data.hands[0]

    myo_samples = np.zeros((1000, 8))
    myo_array = np.array(myo_data[:1000])
    myo_samples[0 : myo_array.shape[0], 0:8] = myo_array

    # https://stackoverflow.com/a/61075207
    # https://stackoverflow.com/a/51926373
    myo_samples = myo_samples.reshape((1, 1000, 8, 1))
    anchor_points = get_anchor_points(hand)

    # https://stackoverflow.com/a/72601148
    prediction = model.predict(myo_samples, verbose=False)
    joint_angles = get_angles_from_prediction(prediction)
    bone_lengths = get_bone_lengths(hand)

    x, y, z = get_points_from_angles(
        anchor_points,
        bone_lengths,
        joint_angles,
    )

    leap_data["input_points"] = leapplot.get_bone_points(hand)
    leap_data["predicted_points"] = np.array([x, y, z])


def leap_collect(callback, leap_data, myo_data):
    class TrackingListener(lp.Listener):
        def __init__(self, callback):
            self.callback = callback

        def on_tracking_event(self, event):
            self.callback(event, leap_data, myo_data)

    leap_connection = lp.Connection()

    leap_connection.set_tracking_mode(lp.TrackingMode.Desktop)
    leap_connection.add_listener(TrackingListener(callback))

    with leap_connection.open():
        leap_connection._poll_loop()


def myo_collect(myo_samples):
    myo = Myo(None, mode=emg_mode.RAW)

    myo.connect()
    myo.add_emg_handler(lambda x, _: myo_samples.append(x))

    running = True

    while running:
        myo.run()


def plot(leap_data):
    fig = plt.figure()

    ax = fig.add_subplot(
        121, projection="3d", xlim=(-300, 300), ylim=(-200, 400), zlim=(-300, 300)
    )

    ax2 = fig.add_subplot(
        122, projection="3d", xlim=(-300, 300), ylim=(-200, 400), zlim=(-300, 300)
    )

    ax.view_init(elev=45.0, azim=122)
    ax2.view_init(elev=45.0, azim=122)

    def animate(frame):
        leapplot.reset_plot(ax)
        leapplot.reset_plot(ax2)

        if not ("input_points" in leap_data) or not ("predicted_points" in leap_data):
            return

        # First plot
        x, y, z = leap_data["input_points"]
        ax.scatter(x, y, z, s=[10] * len(x), alpha=1)

        # Second plot (reconstructed from joint angles)
        x, y, z = leap_data["predicted_points"]
        ax2.scatter(x, y, z, s=[10] * len(x), alpha=1)

    anim = animation.FuncAnimation(fig, animate, blit=False, interval=20)

    plt.show()


if __name__ == "__main__":
    with mp.Manager() as manager:
        rows = manager.list()

        leap_data = manager.dict()
        myo_data = manager.list()

        leap_thread = mp.Process(
            target=leap_collect,
            args=(leap_process_data, leap_data, myo_data),
        )
        myo_thread = mp.Process(target=myo_collect, args=(myo_data,))
        plot_thread = mp.Process(target=plot, args=(leap_data,))

        leap_thread.start()
        myo_thread.start()
        plot_thread.start()

        running = True

        while running:
            time.sleep(0)
