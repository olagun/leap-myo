from leap.events import Event
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
)
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from collections import Counter
import math

sample_rate = 200


def leap_process_data(data, leap_data):
    if not len(data.hands):
        return

    hand = data.hands[0]

    # anchor_points = get_anchor_points(hand)
    joint_angles = get_joint_angles(hand)
    # bone_lengths = get_bone_lengths(hand)

    # x, y, z = get_points_from_angles(
    #     anchor_points,
    #     bone_lengths,
    #     joint_angles,
    # )

    leap_data["joint_angles"] = joint_angles
    leap_data["input_points"] = leapplot.get_bone_points(hand)
    # leap_data["predicted_points"] = np.array([x, y, z])


def leap_collect(callback, leap_data):
    counter = Counter()

    state = {}
    state["start_time"] = time.time()

    class TrackingListener(lp.Listener):
        def __init__(self, callback):
            self.callback = callback

        def on_connection_event(self, _):
            print("leap connected")

        def on_tracking_event(self, event):
            self.callback(event, leap_data)

            counter["samples"] += 1

            if time.time() - state["start_time"] > 1:
                state["samples_per_second"] = counter["samples"]
                state["start_time"] = time.time()
                counter["samples"] = 0

                samples_per_second = state["samples_per_second"]
                print(f"leap samples per second: {samples_per_second}")

    leap_connection = lp.Connection()

    leap_connection.set_tracking_mode(lp.TrackingMode.Desktop)
    leap_connection.add_listener(TrackingListener(callback))

    with leap_connection.open():
        leap_connection._poll_loop()


def myo_collect(myo_samples):
    myo = Myo(None, mode=emg_mode.RAW)

    counter = Counter()

    state = {}
    state["start_time"] = time.time()

    def emg_handler(emg_data, _):
        myo_samples.append(emg_data)

        counter["samples"] += 1

        if time.time() - state["start_time"] > 1:
            state["samples_per_second"] = counter["samples"]
            state["start_time"] = time.time()
            counter["samples"] = 0

            samples_per_second = state["samples_per_second"]
            print(f"myo samples per second: {samples_per_second}")

    myo.connect()
    myo.add_emg_handler(emg_handler)

    running = True

    while running:
        myo.run()


def data_collect(rows, leap_samples, myo_samples):
    data = {}

    running = True

    while running:
        data["time"] = time.time()

        # Leap DataFrame
        if not ("joint_angles" in leap_samples):
            continue

        for d, angles in leap_samples["joint_angles"].items():
            for a in angles:
                data[f"{d}_{a}"] = math.degrees(leap_samples["joint_angles"][d][a])

        # Myo DataFrame
        if not len(myo_samples):
            continue

        myo_sample = myo_samples[-1]

        for i in range(0, len(myo_sample)):
            data[f"channel_{i + 1}"] = myo_sample[i]

        rows.append(data)


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
        try:
            rows = manager.list()

            leap_data = manager.dict()
            myo_data = manager.list()

            leap_thread = mp.Process(
                target=leap_collect,
                args=(
                    leap_process_data,
                    leap_data,
                ),
            )
            myo_thread = mp.Process(target=myo_collect, args=(myo_data,))

            data_thread = mp.Process(
                target=data_collect, args=(rows, leap_data, myo_data)
            )

            leap_thread.start()
            myo_thread.start()
            data_thread.start()

            # Reduces sampling rate
            # plot_thread = mp.Process(target=plot, args=(leap_data,))
            # plot_thread.start()

            running = True

            while running:
                time.sleep(0)

        finally:
            df = pd.DataFrame(list(rows))
            df.to_csv("data.csv", index=False)
