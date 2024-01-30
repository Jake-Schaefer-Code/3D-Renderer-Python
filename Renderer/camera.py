import numpy as np
import math
import pygame
import random
from collections import deque
import cProfile,pstats
import cython
import numba

from quaternions import *
from functions import *
from constants import *


class Camera:
    def __init__(self):
        self.pos = np.array([0,0,0,1])
        #self.lightdir = np.array([0,0,-1,0],dtype='float64')
        self.planes = PLANES
        #self.plane_keys = PLANES.keys()
        self.fovx = FOVX
        self.fovy = FOVY
        self.third_person = False

        # Distance to near (projection) plane
        self.nearz = NEAR_Z
        self.lbn = NEAR_Z * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), 1.0])
        
        # Distance to far plane
        self.farz = FAR_Z
        self.rtf = NEAR_Z * np.array([math.tan(FOVX/2), math.tan(FOVY/2), FAR_Z/NEAR_Z])

        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = self.z_scale(np.array([0,0,0,1]))
        self.center3 = self.z_scale(np.array([0,0,0,1]))
        
        # The matrix that will be modified through transitions and rotations
        self.matrix = np.identity(4)

        # Initializing perspective projection matrix
        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l),      0,          0,       -n*(r+l)/(r-l)],
                                [    0,      2*n/(t-b),      0,       -n*(t+b)/(t-b)],
                                [    0,          0,      (n+f)/(f-n),  2*(n*f)/(n-f)],
                                [    0,          0,          1,              0      ]])
        
        # For future ray tracing
        self.shapes = []
        self.dist_to_nearest = 1
        self.draw_mesh = []

    def update(self) -> None:
        self.nearz = WIDTH/(2*math.tan(self.fovx/2))
        self.planes = [np.array([-self.nearz,    0,      HALFWIDTH,  0]) , # right
                       np.array([ self.nearz,    0,      HALFWIDTH,  0]) , # left
                       np.array([     0,    -self.nearz, HALFHEIGHT, 0]) , # bottom
                       np.array([     0,     self.nearz, HALFHEIGHT, 0])]  # top
        self.planes = np.array([plane/math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2) for plane in self.planes])
        self.fovy = 2 * math.atan(ASPECT_RATIO * math.tan(self.fovx/2))
        self.lbn = self.nearz * np.array([ -math.tan(self.fovx/2),  -math.tan(self.fovy/2),       1.0       ])
        self.rtf = self.nearz * np.array([  math.tan(self.fovx/2),   math.tan(self.fovy/2), FAR_Z/self.nearz])
        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l),      0,          0,       -n*(r+l)/(r-l)],
                                [    0,      2*n/(t-b),      0,       -n*(t+b)/(t-b)],
                                [    0,          0,      (n+f)/(f-n),  2*(n*f)/(n-f)],
                                [    0,          0,          1,              0      ]])
        
    # Applying transformation matrix
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        #return matmul2(self.matrix, point)
        return self.matrix @ point
    
    
    # Projecting point onto projection plane
    def perspective_projection(self, point: np.ndarray) -> np.ndarray:
        persp_point = self.perspM @ point
        return persp_point / (persp_point[3] if persp_point[3] != 0 else 1e-6)

    # Camera movement
    def move_cam(self, trans_vector: list) -> None:
        """if self.key_move: self.center3 = translate(np.identity(4), trans_vector) @ self.center3"""
        self.matrix = translate(self.matrix, trans_vector)
    
    # Camera Rotation
    def rotate_cam(self, axis: np.ndarray, angle: float) -> None:
        # For third person rotation
        if self.third_person:
            self.move_cam(self.center3)
            self.matrix = q_mat_rot(self.matrix, axis, -angle)
            self.move_cam(-self.center3)
        # First person
        else:
            self.matrix = q_mat_rot(self.matrix, axis, angle)
        
    # converts to canonical viewing space
    def z_scale(self, point: np.ndarray) -> np.ndarray:
        n = self.nearz
        f = self.farz
        c1 = 2*f*n/(n-f)
        c2 = (f+n)/(f-n)
        z = c1/(point[2]-c2)
        x = point[0]
        y = -point[1]
        return np.array([x,y,z,point[3]])

"""@numba.njit
def matmul2(matrix, point):
    print(matrix)
    for row in matrix:
        print(row)
    point2 = np.array([np.dot(row, point) for row in matrix])
    return point2"""
        