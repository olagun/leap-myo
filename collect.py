from myo_api import Myo, emg_mode
from leap_api import TrackingListener
import leap as lp
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d as plt3d
import plot as leapplot
import threading
from utils import Ry, Rx, Rz
import time

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

    hand = leap_value.hands[0]
    points = leapplot.get_bone_points(hand)

    # First plot
    ax.scatter(
        points[0],
        points[1],
        points[2],
        s=[10] * len(points[0]),
        alpha=1,
    )

    # Second Plot (Reconstructed from joint angles)
    x = []
    y = []
    z = []

    anchor_points = {}
    bone_lengths = {}
    joint_angles = {}

    digit_labels = ["thumb", "index", "middle", "ring", "pinky"]

    # Thumb, Index, Middle, Ring, Pinky
    for d in range(0, 5):
        digit = hand.digits[d]

        bone_lengths[digit_labels[d]] = {}
        joint_angles[digit_labels[d]] = {}
        anchor_points[digit_labels[d]] = {}

        # Metacarpal, Proximal, Intermediate, Distal
        for b in range(0, 4):
            bone = digit.bones[b]

            prev_joint = np.array(
                [bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z]
            )

            anchor_points[digit_labels[d]][b] = prev_joint

            next_joint = np.array(
                [bone.next_joint.x, bone.next_joint.y, bone.next_joint.z]
            )

            to_from = next_joint - prev_joint
            bone_lengths[digit_labels[d]][b] = np.linalg.norm(to_from)
            to_from = np.array([-to_from[0], to_from[2], to_from[1]])

            aa_angle = -math.atan2(to_from[0], -to_from[1])
            fe_angle = math.atan2(to_from[2], -to_from[1])

            # if digit_labels[d] == "index" and b == 1:
            #     ax2.text(0, 0, 0, f"fe angle: {fe_angle * 180 / math.pi}", size=10)
            #     ax2.text(0, 40, 0, f"aa angle: {aa_angle * 180 / math.pi}", size=10)

            # line = plt3d.art3d.Line3D([0, to_from[0]], [0, to_from[1]], [0, to_from[2]])
            # ax2.add_line(line)

            # Thumb
            if d == 0:
                # Trapeziometacarpal
                if b == 1:
                    joint_angles["thumb"]["tm,f/e"] = fe_angle
                    joint_angles["thumb"]["tm,aa"] = aa_angle
                # Metacarpophalangeal
                elif b == 2:
                    joint_angles["thumb"]["mcp,f/e"] = fe_angle
                    joint_angles["thumb"]["mcp,aa"] = aa_angle
            # Other fingers
            else:
                # Metacarpophalangeal
                if b == 1:
                    joint_angles[digit_labels[d]]["mcp,f/e"] = fe_angle
                    joint_angles[digit_labels[d]]["mcp,aa"] = aa_angle
                # Proximal Interphalangeal
                elif b == 2:
                    joint_angles[digit_labels[d]]["pip"] = fe_angle

    x.append(0)
    y.append(0)
    z.append(0)

    value = {}

    # Reconstruct hand from joint angles
    for digit, angle in joint_angles.items():
        if digit == "thumb":
            value[f"{digit}_{angle["tm,f/e"]}"] = angle["tm,f/e"]
            value[f"{digit}_{angle["tm,aa"]}"] = angle["tm,aa"]

            value[f"{digit}_{angle["mcp,f/e"]}"] = angle["mcp,f/e"]
            value[f"{digit}_{angle["mcp,aa"]}"] = angle["mcp,aa"]
        else:
            value[f"{digit}_{angle["mcp,f/e"]}"] = angle["mcp,f/e"]
            value[f"{digit}_{angle["mcp,aa"]}"] = angle["mcp,aa"]
            
            value[f"{digit}_{angle["pip"]}"] = angle["pip"]

        if digit == "thumb":
            anchor_point = np.array(
                [
                    [-anchor_points[digit][1][0]],
                    [anchor_points[digit][1][2]],
                    [anchor_points[digit][1][1]],
                ]
            )

            x.append(anchor_point[0, 0])
            y.append(anchor_point[1, 0])
            z.append(anchor_point[2, 0])

            # Trapeziometacarapl
            fe_angle = angle["tm,f/e"]
            aa_angle = angle["tm,aa"]

            digit_vector = np.array([[0], [-bone_lengths[digit_labels[d]][1]], [0]])
            to_from = anchor_point + np.dot(
                Rz(-aa_angle), np.dot(Rx(-fe_angle), digit_vector)
            )

            x.append(to_from[0, 0])
            y.append(to_from[1, 0])
            z.append(to_from[2, 0])

            # Metacarpophalangeal
            fe_angle = angle["mcp,f/e"]
            aa_angle = angle["mcp,aa"]

            norm_vector = to_from - anchor_point
            norm_vector /= np.linalg.norm(norm_vector)
            norm_vector = np.dot(Rz(-aa_angle), np.dot(Rx(-fe_angle), norm_vector))
            norm_vector *= bone_lengths[digit_labels[d]][2]
            norm_vector += to_from

            x.append(norm_vector[0, 0])
            y.append(norm_vector[1, 0])
            z.append(norm_vector[2, 0])

        if digit != "thumb":
            anchor_point = np.array(
                [
                    [-anchor_points[digit][1][0]],
                    [anchor_points[digit][1][2]],
                    [anchor_points[digit][1][1]],
                ]
            )

            x.append(anchor_point[0, 0])
            y.append(anchor_point[1, 0])
            z.append(anchor_point[2, 0])

            # Metacarpophalangeal
            fe_angle = angle["mcp,f/e"]
            aa_angle = angle["mcp,aa"]

            digit_vector = np.array([[0], [-bone_lengths[digit_labels[d]][1]], [0]])
            to_from = anchor_point + np.dot(
                Rz(-aa_angle), np.dot(Rx(-fe_angle), digit_vector)
            )

            x.append(to_from[0, 0])
            y.append(to_from[1, 0])
            z.append(to_from[2, 0])

            # Proximal Interphalangeal
            fe_angle = angle["pip"]

            norm_vector = to_from - anchor_point
            norm_vector /= np.linalg.norm(norm_vector)
            norm_vector = Rx(-fe_angle) @ norm_vector
            norm_vector *= bone_lengths[digit_labels[d]][2]
            norm_vector += to_from

            x.append(norm_vector[0, 0])
            y.append(norm_vector[1, 0])
            z.append(norm_vector[2, 0])

            # # Distal Interphalangeal
            # angle = math.acos(
            #     norm_vector.dot(to_from)
            #     / (np.linalg.norm(to_from) * np.linalg.norm(norm_vector))
            # )

            # fe_angle = 2 / 3 * angle
            # norm_vector = norm_vector -

    ax2.scatter(x, y, z, s=[10] * len(x), alpha=1)

    leap_values.append(value)


def on_myo_tracking_event(value, moving):
    myo_values.append(value)


def on_leap_tracking_event(event):
    leap_values.append(event)


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

    while True:
        myo.run()


def data_collect():
    start_time = time.time()
    running = True

    while running:
        curr_time = time.time()

        if curr_time - start_time > 1 / sample_rate:
            data = {
                "timestamp": curr_time,
            }

            myo_value = myo_values[len(myo_values) - 1]
            leap_value = leap_values[len(leap_values) - 1]

            # Myo
            myo_df = pd.DataFrame(myo_value)
            myo_df.columns = [f"channel_{x}" for x in myo_df.columns]

            # Leap
            leap_df = pd.DataFrame(leap_value)
            
            df = pd.concat([myo_df, leap_df])
            
            start_time = time.time()


if __name__ == "__main__":
    try:
        leap_thread = threading.Thread(
            target=leap_collect, args=(on_leap_tracking_event,)
        )
        myo_thread = threading.Thread(target=myo_collect, args=(on_myo_tracking_event,))

        anim = animation.FuncAnimation(fig, animate, blit=False, interval=20)

        leap_thread.start()
        myo_thread.start()

        plt.show()
    finally:
        df.to_csv("data.csv")
