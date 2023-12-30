import numpy as np
import math

def q_mult(q1,q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def q_conjugate(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])


def axisangle_to_q(v, theta):
    v = normalize(v)
    
    x, y, z = v
    theta = theta/ 2
    
    w = math.cos(theta)
    x = x * math.sin(theta)
    y = y * math.sin(theta)
    z = z * math.sin(theta)
    return w, x, y, z

def q_rot(point, axis, angle):
    point = np.roll(point,-3)
    r1 = axisangle_to_q(axis, angle)
    r1c = q_conjugate(r1)
    v = q_mult(q_mult(r1,point),r1c)
    return np.roll(v,-1)

def q_mat_rot(mat, axis, angle):
    r1 = axisangle_to_q(axis, angle)
    #r1c = q_conjugate(r1)
    #qmat = np.array([mat[0][0],-mat[0][1],-mat[0][2],-mat[0][3]])
    
    rmat = make_q_mat(r1)

    M = rmat @ mat
    return M


def make_q_mat(q):
    w,x,y,z = q
    """mat = np.array([[w,-x,-y,-z],
                     [x,w,-z,y],
                     [y,z,w,-x],
                     [z,-y,x,w]])"""
    mat = np.array([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]],dtype='float64')
    mat[0][0] = (2.0 * (w*w + x*x)) - 1.0
    mat[0][1] = 2 * (x*y - w*z)
    mat[0][2] = 2 * (x*z + w*y)

    mat[1][0] = 2 * (x*y + w*z)
    mat[1][1] = 2 * (w*w + y*y) - 1
    mat[1][2] = 2 * (y*z - w*x)

    mat[2][0] = 2 * (x*z - w*y)
    mat[2][1] = 2 * (y*z + w*x)
    mat[2][2] = 2 * (w*w + z*z) -1

    mat[3][3] = 1
    return mat