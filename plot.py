import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
from utils import (
    get_angles,
    get_rot_from_angles,
    quaternion_rotation_matrix,
    get_angles_from_rot,
    euler_from_quaternion,
)

# Leap Motion Hand Animation
finger_bones = ["metacarpals", "proximal", "intermediate", "distal"]


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


def plot_points(points, scatter):
    scatter.set_offsets(points[:2].T)
    scatter.set_3d_properties(points[2], zdir="z")


def get_points(controller):
    """
    Returns points for a simple hand model.
    18 point model. Finger tips + Palm
    """
    frame = controller.frame()
    hand = frame.hands.rightmost
    if not hand.is_valid:
        return None
    fingers = hand.fingers

    X = []
    Y = []
    Z = []

    # Add the position of the palms
    X.append(-1 * hand.palm_position.x)
    Y.append(hand.palm_position.y)
    Z.append(hand.palm_position.z)

    for finger in fingers:
        # Add finger tip positions
        X.append(-1 * finger.tip_position.x)
        Y.append(finger.tip_position.y)
        Z.append(finger.tip_position.z)
    return np.array([X, Z, Y])


def get_rel_points(controller):
    """
    Returns points for a simple hand model.
    Relative to the hand itself.
    18 point model. Finger tips + Palm
    """
    frame = controller.frame()
    hand = frame.hands.rightmost
    if not hand.is_valid:
        return None

    # Transforming finger coordinates into hands POV
    hand_x_basis = hand.basis.x_basis
    hand_y_basis = hand.basis.y_basis
    hand_z_basis = hand.basis.z_basis
    hand_origin = hand.palm_position
    hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
    hand_transform = hand_transform.rigid_inverse()

    fingers = hand.fingers

    X = []
    Y = []
    Z = []

    # Add the position of the palms
    palm_pos = hand_transform.transform_point(hand.palm_position)
    X.append(-1 * palm_pos.x)
    Y.append(palm_pos.y)
    Z.append(palm_pos.z)

    for finger in fingers:
        transformed_position = hand_transform.transform_point(finger.tip_position)
        # Add finger tip positions
        X.append(-1 * transformed_position.x)
        Y.append(transformed_position.y)
        Z.append(transformed_position.z)
    return np.array([X, Z, Y])


def get_stable_points(controller):
    """
    Returns points for a simple hand model.
    18 point model. Finger tips + Palm
    Uses stabalized positions, not reccomended for ML.
    """
    frame = controller.frame()
    hand = frame.hands.rightmost
    if not hand.is_valid:
        return None
    fingers = hand.fingers

    X = []
    Y = []
    Z = []

    # Add the position of the palms
    X.append(-1 * hand.palm_position.x)
    Y.append(hand.palm_position.y)
    Z.append(hand.palm_position.z)

    for finger in fingers:
        # Add finger tip positions
        X.append(-1 * finger.stabilized_tip_position.x)
        Y.append(finger.stabilized_tip_position.y)
        Z.append(finger.stabilized_tip_position.z)
    return np.array([X, Z, Y])


def plot_points(points, scatter):
    scatter.set_offsets(points[:2].T)
    scatter.set_3d_properties(points[2], zdir="z")


def plot_simple(points, ax):
    """
    Plot lines connecting the palms to the fingers, assuming thats the only data we get.
    Assumes we are using the 18 point tip model from get_points().
    """
    # Get Palm Position
    palm = points[:, 0]

    # For Each of the 5 fingers
    for n in range(1, 6):
        # Draw a line from the palm to the finger tips
        tip = points[:, n]
        top = plt3d.art3d.Line3D(
            [palm[0], tip[0]], [palm[1], tip[1]], [palm[2], tip[2]]
        )
        ax.add_line(top)


def reset_plot(ax):
    """
    The Line plots will plot other eachother, as I make new lines instead of changing the data for the old ones
    TODO: Fix plot_simple and plot_lines so I don't need to do this.
    """
    # Reset the plot
    ax.cla()
    # Really you can just update the lines to avoid this
    ax.set_xlim3d([-200, 200])
    ax.set_xlabel("X [mm]")
    ax.set_ylim3d([-200, 150])
    ax.set_ylabel("Y [mm]")
    ax.set_zlim3d([-100, 300])
    ax.set_zlabel("Z [mm]")


# Plotting the whole hand
def plot_bone_lines(points, ax):
    """
    Plot the lines for the hand based on a full hand model.
    (22 points, 66 vars)
    """
    mcps = []

    # For Each of the 5 fingers
    for i in range(0, 5):
        n = 4 * i

        # Get each of the bones
        mcp = points[:, n + 0]
        pip = points[:, n + 1]
        dip = points[:, n + 2]
        tip = points[:, n + 3]

        # Connect the lowest joint to the middle joint
        bot = plt3d.art3d.Line3D([mcp[0], pip[0]], [mcp[1], pip[1]], [mcp[2], pip[2]])
        ax.add_line(bot)

        # Connect the middle joint to the top joint
        mid = plt3d.art3d.Line3D([pip[0], dip[0]], [pip[1], dip[1]], [pip[2], dip[2]])
        ax.add_line(mid)

        # Connect the top joint to the tip of the finger
        top = plt3d.art3d.Line3D([dip[0], tip[0]], [dip[1], tip[1]], [dip[2], tip[2]])
        ax.add_line(top)

        # Connect each of the fingers together
        mcps.append(mcp)

    for mcp in range(0, 4):
        line = plt3d.art3d.Line3D(
            [mcps[mcp][0], mcps[mcp + 1][0]],
            [mcps[mcp][1], mcps[mcp + 1][1]],
            [mcps[mcp][2], mcps[mcp + 1][2]],
        )
        ax.add_line(line)

    # Connext the left hand side of the index finger to the thumb.
    thumb_mcp = points[:, 1 + 2]
    pinky_mcp = points[:, 4 + 2]
    line = plt3d.art3d.Line3D(
        [thumb_mcp[0], pinky_mcp[0]],
        [thumb_mcp[1], pinky_mcp[1]],
        [thumb_mcp[2], pinky_mcp[2]],
    )
    ax.add_line(line)


def get_bone_points(hand):
    """
    Uses 4 joints for each finger, 3 for the thumb
    """
    X = []
    Y = []
    Z = []

    # thumb, index, middle, ring, pinky
    for digit_index in range(0, 5):
        digit = hand.digits[digit_index]

        # metacarpal, proximal, intermediate, distal
        for bone_index in range(0, 4):
            """
            0 = metacarpophalangeal joint, or knuckle, of the finger.
            1 = proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
            2 = distal interphalangeal joint of the finger. This joint is closest to the tip.
            3 = tip of the finger.
            """

            bone = digit.bones[bone_index]

            X.append(-1 * bone.prev_joint[0])
            Y.append(bone.prev_joint[1])
            Z.append(bone.prev_joint[2])

    return np.array([X, Z, Y])


def get_joint_angles(hand):
    """
    Returns 16 joint angles from a hand
    """
    joint_angles = []

    for d in range(0, 5):
        # Thumb
        if d == 0:
            tm = hand.digits[d].proximal
            mcp = hand.digits[d].intermediate

            joint_angles.append(())
        else:
            # Other fingers
            pass


def get_rel_bone_points(controller):
    """
    Gets points for a full hand model. (22 points, 66 vars)
    Relative to the hand itself.
    Uses 4 joints for each finger and 3 for the thumb.
    Also uses Palm and Wrist position.
    Note this could be reduced to 21 points as the thumb has 1 less joint.
    """
    frame = controller.frame()
    hand = frame.hands.rightmost
    if not hand.is_valid:
        return None

    # Get hand transform
    hand_x_basis = hand.basis.x_basis
    hand_y_basis = hand.basis.y_basis
    hand_z_basis = hand.basis.z_basis
    hand_origin = hand.palm_position
    hand_transform = Leap.Matrix(hand_x_basis, hand_y_basis, hand_z_basis, hand_origin)
    hand_transform = hand_transform.rigid_inverse()

    fingers = hand.fingers

    X = []
    Y = []
    Z = []

    # Add the position of the palms
    # Transform palm position
    palm_pos = hand_transform.transform_point(hand.palm_position)
    X.append(palm_pos.x)
    Y.append(palm_pos.y)
    Z.append(palm_pos.z)

    # Add wrist position
    wrist_pos = hand_transform.transform_point(hand.wrist_position)
    X.append(wrist_pos.x)
    Y.append(wrist_pos.y)
    Z.append(wrist_pos.z)

    # Add fingers
    for finger in fingers:
        for joint in range(0, 4):
            """
            0 = JOINT_MCP – The metacarpophalangeal joint, or knuckle, of the finger.
            1 = JOINT_PIP – The proximal interphalangeal joint of the finger. This joint is the middle joint of a finger.
            2 = JOINT_DIP – The distal interphalangeal joint of the finger. This joint is closest to the tip.
            3 = JOINT_TIP – The tip of the finger.
            """
            # Transform the finger
            transformed_position = hand_transform.transform_point(
                finger.joint_position(joint)
            )
            X.append(transformed_position[0])
            Y.append(transformed_position[1])
            Z.append(transformed_position[2])

    return np.array([X, Z, Y])


# Other Leap Funcs
def save_points(points, name="points.csv"):
    # Save one single row/frame to disk
    np.savetxt(name, points, delimiter=",")
