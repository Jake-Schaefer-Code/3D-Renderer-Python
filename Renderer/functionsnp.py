import numpy as np
import math
from constants import *
import time
import random
import numba

slope = None

@numba.njit
def signed_distance(plane_normal, vertex, plane_point=[0,0,0,0]):
    return np.dot(plane_normal[:3], (vertex[:3] - plane_point[:3]))


@numba.njit
def line_plane_intersect(plane_normal: np.ndarray, p1: np.ndarray, p2: np.ndarray, plane_point: np.ndarray=np.array([0,0,0,0])) -> np.ndarray:
    p1x,p1y,p1z,p1w = p1
    p2x,p2y,p2z,p2w = p2
    # Normal values
    nx, ny, nz, nw = plane_normal
    slopex, slopey, slopez = p1x-p2x,p1y-p2y,p1z-p2z
    #slope = [p1x-p2x,p1y-p2y,p1z-p2z,0]
    #slope = p1-p2
    #mag = ((slope[0]**2) + (slope[1]**2) + (slope[2]**2)) ** 0.5
    mag = ((slopex**2) + (slopey**2) + (slopez**2)) ** 0.5
    if mag == 0: mag = 1e-6
    #unit_slope = [slope[0]/mag, slope[1]/mag, slope[2]/mag]
    ux, uy, uz = slopex/mag, slopey/mag, slopez/mag
    #unit_slope = slope / (mag if mag != 0 else 1e-6)
    # parametric value
    #dot = unit_slope[0] * plane_normal[0] + unit_slope[1] * plane_normal[1] +  unit_slope[2] * plane_normal[2]
    #dot_top = point[0] * plane_normal[0] + point[1] * plane_normal[1] + point[2] * plane_normal[2]
    #dot = np.dot(unit_slope, plane_normal)
    #dot = unit_slope[0] * plane_normal[0] + unit_slope[1] * plane_normal[1] + unit_slope[2] * plane_normal[2]
    #dot = unit_slope[0] * nx + unit_slope[1] * ny + unit_slope[2] * nz
    dot = ux * nx + uy * ny + uz * nz
    
    if dot == 0: dot = 1e-6
    #d = np.dot((plane_point - p2), plane_normal) / (dot if dot != 0 else 1e-6)
    d = (plane_point[0] - p2x) * nx + (plane_point[1] - p2y) * ny + (plane_point[2] - p2z) * nz
    """if dot == 0:
        d = np.dot((plane_point - p2), plane_normal) / 0.000001
    else:
        d = np.dot((plane_point - p2), plane_normal) / dot"""
    return np.array([p2x + ux * d, p2y + uy * d, p2z + uz * d,1])
    #return p2 + unit_slope * d


def get_face_normal(face):
    # Note: this is if the vertices are in clockwise order. If they are not, then the cross product should be the other way
    x1, y1, z1, w1 = (face[2] - face[0])
    x2, y2, z2, w2 = (face[1] - face[0])
    # returns the vector perpendicular to the face, which I will compare with either the camera direction or the light direction
    normal = [(y1 * z2 - z1 * y2), 
              (z1 * x2 - x1 * z2), 
              (x1 * y2 - y1 * x2), 
               0]
    nx, ny, nz, nw = normal
    norm = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    if norm != 0:
        normal = [nx/norm, ny/norm, nz/norm, 0]
    return normal

@numba.njit
def to_pygame(point):
    #point = [(point[0] * WIDTH/2) + WIDTH/2, (point[1] * HEIGHT/2) + HEIGHT/2]
    #return [(point[0]+1)*WIDTH/2, (point[1]+1)*HEIGHT/2]
    return [(point[0] * WIDTH/2) + WIDTH/2, (point[1] * HEIGHT/2) + HEIGHT/2]

def length(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)


def translate(matrix, trans_vector):
    return np.array([[1,0,0,-trans_vector[0]],
                     [0,1,0,-trans_vector[1]],
                     [0,0,1,-trans_vector[2]],
                     [0,0,0,1 ]])  @ matrix

def to_screen_space(point):
    return point * np.array([ASPECT_RATIO/math.tan(FOVX/2),1/math.tan(FOVY/2),1,1])

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))


"""persp_point = np.array([perspM[0][0] * point[0] + perspM[0][1] * point[1] + perspM[0][2] * point[2] + perspM[0][3] * point[3],
                                perspM[1][0] * point[0] + perspM[1][1] * point[1] + perspM[1][2] * point[2] + perspM[1][3] * point[3],
                                perspM[2][0] * point[0] + perspM[2][1] * point[1] + perspM[2][2] * point[2] + perspM[2][3] * point[3],
                                perspM[3][0] * point[0] + perspM[3][1] * point[1] + perspM[3][2] * point[2] + perspM[3][3] * point[3]

        ])"""