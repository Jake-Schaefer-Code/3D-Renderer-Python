import numpy as np
import math
from scipy.spatial.transform import Rotation as R

QUATERNION_MATRIX = np.eye(4,4)

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
    #mag = np.linalg.norm(v)
    if mag != 0:
        v = v / mag
    #return np.array([v[0]/mag,v[1]/mag,v[2]/mag])
    return v

def conjugate(q):
    return [q[0],-q[1],-q[2],-q[3]]

def invert_quaternion(q):
    w, x, y, z = q
    denom = w**2 + x**2 + y**2 +z**2
    return [w/denom, -x/denom, -y/denom, -z/denom]

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
    #v = quaternion_multiplication(r1,point)
    v = quaternion_multiplication(v,r1_c)
    #v = quaternion_multiplication(r1_c,v)
    # turns the vector back into the correct orientation for our homogenous coordinate system
    return np.roll(v,-1)

# This is used instead of q_rot to update projection matrix in 3D renderer 
def q_mat_rot(mat: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    q = rotation_quaternion(axis, angle)
    
    """sr1 = R.from_quat(np.roll(r1,-1))
    sr1_inv = sr1.inv()
    matrix = make_q_mat(r1)
    inv_matrix = make_q_mat(r2)
    print(np.append(sr1_inv.apply([1,0,0]),0))
    print(inv_matrix @ np.array([1,0,0,0]))
    print()"""

    return make_q_mat(q) @ mat


def make_q_mat(q) -> np.ndarray:
    w,x,y,z = q
    """QUATERNION_MATRIX[0][0] = -2 * (w * w + x * x) + 1
    QUATERNION_MATRIX[0][1] = 2 * (x * y - w * z)
    QUATERNION_MATRIX[0][2] = 2 * (x * z + w * y)
    QUATERNION_MATRIX[1][0] = 2 * (x * y + w * z)
    QUATERNION_MATRIX[1][1] = -2 * (w * w + y * y) + 1
    QUATERNION_MATRIX[1][2] = 2 * (y * z - w * x)
    QUATERNION_MATRIX[2][0] = 2 * (x * z - w * y)
    QUATERNION_MATRIX[2][1] = 2 * (y * z + w * x)
    QUATERNION_MATRIX[2][2] = -2 * (w * w + z * z) +1"""
    # Quaternion rotation matrix
    #return QUATERNION_MATRIX
    return np.array([[2 * (w * w + x * x) - 1,  2 * (x * y - w * z),     2 * (x * z + w * y),     0],
                     [2 * (x * y + w * z),      2 * (w * w + y * y) - 1, 2 * (y * z - w * x),     0],
                     [2 * (x * z - w * y),      2 * (y * z + w * x),     2 * (w * w + z * z) - 1, 0],
                     [0,                        0,                       0,                       1]])


"""def main():
    mat = np.eye(4,4)
    q_mat_rot(mat, np.array([1,1,1]), -math.pi/45)
    q_mat_rot(mat, np.array([0,1,1]), -math.pi/45)
    q_mat_rot(mat, np.array([1,0,1]), -math.pi/45)
    


if __name__ == '__main__':
    main()"""