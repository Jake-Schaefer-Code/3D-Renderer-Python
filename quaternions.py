import numpy as np
import math

def quaternion_multiplication(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # parentheses for neatness
    return np.array([(w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2), 
                     (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2), 
                     (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2), 
                     (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2)])   

def normalize_vector(v: np.ndarray) -> np.ndarray:
    mag = ((v[0]**2) + (v[1]**2) + (v[2]**2)) ** 0.5
    if mag == 0: mag = 1e-6
    if mag != 0:
        v = v / mag
    return v

def conjugate(q: list):
    return [q[0],-q[1],-q[2],-q[3]]

# vector to rotate by
def rotation_quaternion(v: np.ndarray, theta: float) -> list:
    v = normalize_vector(v)
    x, y, z = v * math.sin(theta / 2)
    w = math.cos(theta / 2)
    return [w, x, y, z]

# not really used
def q_rot(point, axis, angle):
    point = np.roll(point,-3)
    r1 = rotation_quaternion(axis, angle)
    r1_c = conjugate(r1)
    # rotates the point with the 
    v = quaternion_multiplication(point,r1)
    v = quaternion_multiplication(v,r1_c)
    # turns the vector back into the correct orientation for our homogenous coordinate system
    return np.roll(v,-1)

# This is used instead of q_rot to update projection matrix in 3D renderer 
def q_mat_rot(mat: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    q = rotation_quaternion(axis, angle)
    return make_q_mat(q) @ mat


def make_q_mat(q: list) -> np.ndarray:
    w,x,y,z = q
    # Quaternion rotation matrix
    return np.array([[2 * (w * w + x * x) - 1,  2 * (x * y - w * z),     2 * (x * z + w * y),     0],
                     [2 * (x * y + w * z),      2 * (w * w + y * y) - 1, 2 * (y * z - w * x),     0],
                     [2 * (x * z - w * y),      2 * (y * z + w * x),     2 * (w * w + z * z) - 1, 0],
                     [0,                        0,                       0,                       1]])