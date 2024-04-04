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

    if not len(leap_values):
        return

    leap_value = leap_values[len(leap_values) - 1]

    if len(leap_value.hands) == 0:
        return

    points = leapplot.get_bone_points(leap_value.hands[0])
    a_points = points

    patches = ax.scatter(points[0], points[1], points[2], s=[10] * 21, alpha=1)
    leapplot.plot_points(points, patches)
    leapplot.plot_bone_lines(points, ax)

    hand = leap_value.hands[0]
    angles = np.array(get_angles(hand))

    X = [0]
    Y = [0]
    Z = [0]

    for finger in range(0, 5):
        for bone in range(0, 3):
            pitch = angles[finger, bone, 0]
            yaw = angles[finger, bone, 1]
            roll = angles[finger, bone, 2]

            theta = angles[finger, bone, :]

            # theta = [pitch, yaw, roll]
            rot_mat = get_rot_from_angles(theta)

            # Which basis is this bone defined in???
            b = leap_value.hands[0].digits[finger].bones[bone]
            bone_len = np.linalg.norm(
                np.array([b.prev_joint.x, b.prev_joint.y, b.prev_joint.z])
                - np.array([b.next_joint.x, b.next_joint.y, b.next_joint.z])
            )
            new_bone = rot_mat.dot(np.array([0, bone_len, 0]))

            x = X[finger * 3 + bone] + new_bone[0]
            y = Y[finger * 3 + bone] + new_bone[1]
            z = Z[finger * 3 + bone] + new_bone[2]

            X.append(x)
            Y.append(y)
            Z.append(z)

    # Convert to a numpy array
    a_points = [X, Z, Y]
    a_points = np.array(a_points)

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
