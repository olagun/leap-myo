import numpy as np
import math

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
