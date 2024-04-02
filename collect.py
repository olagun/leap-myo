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
import Matplotleap as leapplot
import threading

sample_rate = 200

myo_values = []
leap_values = []

df = pd.DataFrame()

NUM_POINTS = 30

# Matplotlib Setup
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


# Angle Utils
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

    return rot_matrix


def get_angles_from_rot(rot_mat):
    """
    Function from LearnOpenCV, Satya Mallick:
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    https://github.com/spmallick/learnopencv/blob/master/RotationMatrixToEulerAngles/rotm2euler.py
    """
    sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
        y = math.atan2(-rot_mat[2, 0], sy)
        z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = math.atan2(-rot_mat[2, 0], sy)
        z = 0

    return [x, y, z]


def get_rot_from_angles(theta):
    # Calculates Rotation Matrix given euler angles.
    """
    Function from LearnOpenCV, Satya Mallick:
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    https://github.com/spmallick/learnopencv/blob/master/RotationMatrixToEulerAngles/rotm2euler.py
    """
    x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(z, np.dot(y, x))
    return R


finger_bones = ["metacarpals", "proximal", "intermediate", "distal"]


def get_angles(hand):
    """
    Gets angles in degrees for all joints in the hand.
    Do I need the basis vector for the hands?
    """
    angles = []

    for finger in hand.digits:
        bone_angles = []

        for b in range(1, 4):
            last_bone = finger.bones[b - 1]
            curr_bone = finger.bones[b]

            # Generate rotation matrices from basis vectors
            last_bone_mat = quaternion_rotation_matrix(last_bone.rotation)
            curr_bone_mat = quaternion_rotation_matrix(curr_bone.rotation)

            # Get rotation matrix between bones, change of basis
            rot_mat = np.matmul(curr_bone_mat, last_bone_mat.transpose())

            # Generate euler angles in degrees from rotation matrix
            bone_angles.append(get_angles_from_rot(rot_mat))

        angles.append(bone_angles)

    return angles


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
            theta = angles[finger, bone, :]

            print(finger, bone, theta * 180 / math.pi)

            # theta = [pitch, yaw, roll]
            rot_mat = get_rot_from_angles(theta)
            # Which basis is this bone defined in???
            bone_assume = np.array([0, 20, 0])
            # .dot(get_rot_from_angles(angles[0, bone, :]))
            new_bone = rot_mat.dot(bone_assume)

            # Debugging
            if finger == 1:
                if bone == 1:
                    print("Pitch degrees", theta[0] * 57.296)
                    print("Angles", theta)
                    print("rot_mat ", rot_mat)
                    print("Det", np.linalg.det(rot_mat))
                    # Testing time
                    print("nb", new_bone)

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


def data_collect():
    last_time = time.time()

    while True:
        curr_time = time.time()

        myo_value = myo_values[len(myo_values) - 1]
        leap_value = leap_values[len(leap_values) - 1]

        if not myo_value or not leap_value:
            continue

        if curr_time - last_time > 1 / sample_rate:
            # EMG Data
            data = {}
            for i in range(myo_value):
                data["channel_" + i] = myo_value[i]

            # Leap Data

            # df.insert()

            last_time = curr_time


if __name__ == "__main__":
    leap_thread = threading.Thread(target=leap_collect, args=(on_leap_tracking_event,))
    myo_thread = threading.Thread(target=myo_collect, args=(on_myo_tracking_event,))
    data_thread = threading.Thread(target=data_collect)

    anim = animation.FuncAnimation(fig, animate, blit=False, interval=20)

    leap_thread.start()
    # myo_thread.start()
    # data_thread.start()

    plt.show()

    # data = zip(timestamps, myo_values, leap_values)
    # df = pd.DataFrame(data=data)
    # df.to_csv("data.csv")
