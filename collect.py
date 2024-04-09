from myo_api import Myo, emg_mode
import leap as lp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import plot as leapplot
import threading
from utils import (
    get_anchor_points,
    get_joint_angles,
    get_bone_lengths,
    get_points_from_angles,
)
import time

# Settings
sample_rate = 100

# Samples
myo_samples = []
leap_samples = []
leap_joint_angle_samples = []

# Matplotlib
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

    # Get hand
    leap_sample = leap_samples[-1] if len(leap_samples) else None

    if not leap_sample or len(leap_sample.hands) == 0:
        return

    hand = leap_sample.hands[0]

    # First plot
    x, y, z = leapplot.get_bone_points(hand)

    ax.scatter(
        x,
        y,
        z,
        s=[10] * len(x),
        alpha=1,
    )

    # Second plot (reconstructed from joint angles)
    anchor_points = get_anchor_points(hand)
    joint_angles = get_joint_angles(hand)
    bone_lengths = get_bone_lengths(hand)

    x, y, z = get_points_from_angles(
        anchor_points,
        bone_lengths,
        joint_angles,
    )

    ax2.scatter(x, y, z, s=[10] * len(x), alpha=1)

    leap_joint_angle_samples.append(joint_angles)


def on_myo_tracking_event(sample, moving):
    myo_samples.append(sample)


def on_leap_tracking_event(event):
    leap_samples.append(event)


def leap_collect(callback):
    class TrackingListener(lp.Listener):
        def __init__(self, callback):
            self.callback = callback

        def on_tracking_event(self, event):
            self.callback(event)

    leap_connection = lp.Connection()

    leap_connection.set_tracking_mode(lp.TrackingMode.Desktop)
    leap_connection.add_listener(TrackingListener(callback))

    with leap_connection.open():
        leap_connection._poll_loop()


def myo_collect(callback):
    myo = Myo(None, mode=emg_mode.RAW)

    myo.connect()
    myo.add_emg_handler(callback)

    running = True

    while running:
        myo.run()


def data_collect(rows):
    running = True

    while running:
        data = {}

        # Leap DataFrame
        leap_joint_angle_sample = (
            leap_joint_angle_samples[-1] if len(leap_joint_angle_samples) else None
        )

        if not leap_joint_angle_sample:
            continue

        for d, angles in leap_joint_angle_sample.items():
            for a in angles:
                data[f"{d}_{a}"] = [leap_joint_angle_sample[d][a]]

        # Myo DataFrame
        # myo_sample = myo_samples[-1] if len(myo_samples) else None

        # if not myo_sample:
        #     continue

        # for i in range(0, len(myo_sample)):
        #     data[f"channel_{i}"] = myo_sample[i]

        data["time"] = time.time()

        rows.append(data)

        time.sleep(1 / sample_rate)


if __name__ == "__main__":
    rows = []

    try:
        leap_thread = threading.Thread(
            target=leap_collect, args=(on_leap_tracking_event,)
        )
        myo_thread = threading.Thread(target=myo_collect, args=(on_myo_tracking_event,))
        data_thread = threading.Thread(target=data_collect, args=(rows,))

        anim = animation.FuncAnimation(fig, animate, blit=False, interval=20)

        leap_thread.start()
        # myo_thread.start()
        data_thread.start()

        plt.show()

    finally:
        df = pd.DataFrame(rows)
        df.to_csv("data.csv")
