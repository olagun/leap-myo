from myo_api import Myo, emg_mode
from leap_api import Canvas, TrackingListener
import leap as lp
import cv2
import time
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
import plot as leapplot
import threading
from utils import get_angles, get_rot_from_angles

# Settings
sample_rate = 200

# Data
myo_values = []
leap_values = []
df = pd.DataFrame()

NUM_POINTS = 30

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

points = np.zeros((3, NUM_POINTS))
a_points = np.zeros((3, 16))
patches = ax.scatter(points[0], points[1], points[2], s=[20] * NUM_POINTS, alpha=1)
angle_plot = ax2.scatter(a_points[0], a_points[1], a_points[2], s=[10] * 16, alpha=1)

finger_bones = ["metacarpals", "proximal", "intermediate", "distal"]


def animate(frame):
    # Reset the plots
    leapplot.reset_plot(ax)
    leapplot.reset_plot(ax2)

    # No values have been collected yet
    if not len(leap_values):
        return

    # Get latest leap value
    leap_value = leap_values[len(leap_values) - 1]

    # Make sure that there's at last one hand detected
    if len(leap_value.hands) == 0:
        return

    points = leapplot.get_bone_points(leap_value.hands[0])
    a_points = points

    patches = ax.scatter(points[0], points[1], points[2], s=[10] * 21, alpha=1)
    leapplot.plot_points(points, patches)
    leapplot.plot_bone_lines(points, ax)

    # Creating the 2nd plot
    angle_plot = ax2.scatter(a_points[0], a_points[1], a_points[2], alpha=1)
    # Plot Angle points
    leapplot.plot_points(a_points, angle_plot)


def on_myo_tracking_event(value, moving):
    myo_values.append(value)


def on_leap_tracking_event(event):
    leap_values.append(event)


class TrackingListener(lp.Listener):
    def __init__(self, callback):
        self.callback = callback

    def on_tracking_event(self, event):
        self.callback(event)


def leap_collect(callback):
    leap_connection = lp.Connection()
    leap_connection.set_tracking_mode(lp.TrackingMode.Desktop)
    leap_connection.add_listener(TrackingListener(callback))

    with leap_connection.open():
        leap_connection._poll_loop()


def myo_collect(callback):
    myo = Myo(None, mode=emg_mode.RAW)
    myo.connect()
    myo.add_emg_handler(callback)

    while True:
        myo.run()


if __name__ == "__main__":
    leap_thread = threading.Thread(target=leap_collect, args=(on_leap_tracking_event,))
    myo_thread = threading.Thread(target=myo_collect, args=(on_myo_tracking_event,))

    anim = animation.FuncAnimation(fig, animate, blit=False, interval=20)

    leap_thread.start()
    # myo_thread.start()
    # data_thread.start()

    plt.show()

    # data = zip(timestamps, myo_values, leap_values)
    # df = pd.DataFrame(data=data)
    # df.to_csv("data.csv")
