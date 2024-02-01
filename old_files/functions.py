import numpy as np
import math
from constants import *
import time

def signed_distance(plane_normal, plane_point, vertex):
    dot = plane_normal[0] * (vertex[0] - plane_point[0]) + plane_normal[1] * (vertex[1] - plane_point[1]) + plane_normal[2] * (vertex[2] - plane_point[2])
    return dot

def line_plane_intersect(plane_normal, plane_point, p1, p2):
    slope = p1 - p2
    mag = math.sqrt((slope[0]**2) + (slope[1]**2) + (slope[2]**2) + (slope[3]**2))
    if mag == 0:
        unit_slope = slope/0.000001
    else:
        unit_slope = slope/mag
    
    # parametric value
    #dot = unit_slope[0] * plane_normal[0] + unit_slope[1] * plane_normal[1] +  unit_slope[2] * plane_normal[2]
    #dot_top = point[0] * plane_normal[0] + point[1] * plane_normal[1] + point[2] * plane_normal[2]
    dot = np.dot(unit_slope, plane_normal)
    if dot == 0:
        d = np.dot((plane_point - p2), plane_normal) / 0.000001
    else:
        d = np.dot((plane_point - p2), plane_normal) / dot
    return p2 + unit_slope * d

def get_face_normal(face):
    # Note: this is if the vertices are in clockwise order. If they are not, then the cross product should be the other way
    x1, y1, z1, w1 = (face[2] - face[0])
    x2, y2, z2, w2 = (face[1] - face[0])
    # returns the vector perpendicular to the face, which I will compare with either the camera direction or the light direction
    normal = [y1 * z2 - z1 * y2, 
                z1 * x2 - x1 * z2, 
                x1 * y2 - y1 * x2, 
                0]
    nx, ny, nz, nw = normal
    norm = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    if norm != 0:
        normal = [nx/norm, ny/norm, nz/norm, 0]
    return normal


def to_pygame(point):
    return ((point * SCREEN_SCALE) + SCREEN_ORIGIN)[:-2]
