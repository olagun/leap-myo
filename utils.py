import numpy as np
import math

import numpy as np
import math as m

import plot as leapplot

digit_labels = ["thumb", "index", "middle", "ring", "pinky"]

bone_labels = ["metacarpal", "proximal", "intermediate", "distal"]


# Reconstruct hand from joint angles
def get_points_from_angles(anchor_points, bone_lengths, joint_angles):
    x = []
    y = []
    z = []

    for digit, angle in joint_angles.items():
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

            digit_vector = np.array([[0], [-bone_lengths[digit][1]], [0]])
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
            norm_vector = np.dot(
                Rz(-aa_angle + angle["tm,aa"]),
                np.dot(Rx(-fe_angle + angle["tm,f/e"]), norm_vector),
            )
            norm_vector *= bone_lengths[digit][2]
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

            digit_vector = np.array([[0], [-bone_lengths[digit][1]], [0]])
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
            norm_vector *= bone_lengths[digit][2]
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

    return x, y, z


def get_bone_lengths(hand):
    bone_lengths = {}

    # Thumb, Index, Middle, Ring, Pinky
    for d in range(0, 5):
        bone_lengths[digit_labels[d]] = {}

        # Metacarpal, Proximal, Intermediate, Distal
        for b in range(0, 4):
            prev_joint = np.array(
                [
                    hand.digits[d].bones[b].prev_joint.x,
                    hand.digits[d].bones[b].prev_joint.y,
                    hand.digits[d].bones[b].prev_joint.z,
                ]
            )

            next_joint = np.array(
                [
                    hand.digits[d].bones[b].next_joint.x,
                    hand.digits[d].bones[b].next_joint.y,
                    hand.digits[d].bones[b].next_joint.z,
                ]
            )

            bone_lengths[digit_labels[d]][b] = np.linalg.norm(next_joint - prev_joint)

    return bone_lengths


def get_joint_angles(hand):
    joint_angles = {}

    # Thumb, Index, Middle, Ring, Pinky
    for d in range(0, 5):
        joint_angles[digit_labels[d]] = {}

        # Metacarpal, Proximal, Intermediate, Distal
        for b in range(0, 4):
            prev_joint = np.array(
                [
                    hand.digits[d].bones[b].prev_joint.x,
                    hand.digits[d].bones[b].prev_joint.y,
                    hand.digits[d].bones[b].prev_joint.z,
                ]
            )

            next_joint = np.array(
                [
                    hand.digits[d].bones[b].next_joint.x,
                    hand.digits[d].bones[b].next_joint.y,
                    hand.digits[d].bones[b].next_joint.z,
                ]
            )

            to_from = next_joint - prev_joint
            to_from = np.array([-to_from[0], to_from[2], to_from[1]])

            aa_angle = -math.atan2(to_from[0], -to_from[1])
            fe_angle = math.atan2(to_from[2], -to_from[1])

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

    return joint_angles


def get_anchor_points(hand):
    anchor_points = {}

    # Thumb, Index, Middle, Ring, Pinky
    for d in range(0, 5):
        anchor_points[digit_labels[d]] = {}

        # Metacarpal, Proximal, Intermediate, Distal
        for b in range(0, 4):
            anchor_points[digit_labels[d]][b] = np.array(
                [
                    hand.digits[d].bones[b].prev_joint.x,
                    hand.digits[d].bones[b].prev_joint.y,
                    hand.digits[d].bones[b].prev_joint.z,
                ]
            )

    return anchor_points


def Rx(theta):
    return np.matrix(
        [[1, 0, 0], [0, m.cos(theta), -m.sin(theta)], [0, m.sin(theta), m.cos(theta)]]
    )


def Ry(theta):
    return np.matrix(
        [[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]]
    )


def Rz(theta):
    return np.matrix(
        [[m.cos(theta), -m.sin(theta), 0], [m.sin(theta), m.cos(theta), 0], [0, 0, 1]]
    )


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


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


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
