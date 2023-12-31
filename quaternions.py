import numpy as np
import math
import time


def quaternion_multiplication(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # parentheses for neatness
    return np.array([(w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2), 
                     (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2), 
                     (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2), 
                     (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2)])   

def normalize_vector(v):
    n = np.linalg.norm(v)
    if n != 0:
        v = v / n
    return v

def conjugate(q):
    q[1:] *= -1
    return q

# vector to rotate by
def rot_vect(v, theta):
    theta = theta / 2
    v = normalize_vector(v)
    x, y, z = v * math.sin(theta)
    w = math.cos(theta)
    return np.array([w, x, y, z])

# not really used
def q_rot(point, axis, angle):
    point = np.roll(point,-3)
    r1 = rot_vect(axis, angle)
    r1_c = conjugate(r1)
    # rotates the point with the 
    v = quaternion_multiplication(point,r1)
    #v = quaternion_multiplication(r1,point)
    v = quaternion_multiplication(v,r1_c)
    #v = quaternion_multiplication(r1_c,v)
    # turns the vector back into the correct orientation for our homogenous coordinate system
    return np.roll(v,-1)

# This is used instead of q_rot to update projection matrix in 3D renderer 
def q_mat_rot(mat, axis, angle):
    r1 = rot_vect(axis, angle)
    #r1c = q_conjugate(r1)
    #qmat = np.array([mat[0][0],-mat[0][1],-mat[0][2],-mat[0][3]])
    
    rmat = make_q_mat(r1)

    M = rmat @ mat
    return M


def make_q_mat(q):
    w,x,y,z = q

    # Quaternion rotation matrix
    mat = np.array([[2 * (w * w + x * x) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
                    [2 * (x * y + w * z), 2 * (w * w + y * y) - 1, 2 * (y * z - w * x), 0],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w * w + z * z) -1, 0],
                    [0, 0, 0, 1]])
    """mat = np.identity(4,dtype='float64')
    mat[0][0] = 2 * (w * w + x * x) - 1
    mat[0][1] = 2 * (x * y - w * z)
    mat[0][2] = 2 * (x * z + w * y)

    mat[1][0] = 2 * (x * y + w * z)
    mat[1][1] = 2 * (w * w + y * y) - 1
    mat[1][2] = 2 * (y * z - w * x)

    mat[2][0] = 2 * (x * z - w * y)
    mat[2][1] = 2 * (y * z + w * x)
    mat[2][2] = 2 * (w * w + z * z) -1"""
    return mat