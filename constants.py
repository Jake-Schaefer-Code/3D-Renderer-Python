import math
import numpy as np
import pygame as pg
import os
from collections import deque
#from matrices import *
"""Constants for 3D Renderer"""

# Desired width and height of screen
SCREEN_WIDTH = 1080    
SCREEN_HEIGHT = 720
HALFWIDTH = SCREEN_WIDTH/2
HALFHEIGHT = SCREEN_HEIGHT/2

# This is because we define the coordinate system such that +1 is up, but pygame defines it opposite
PYGAME_WINDOW_SCALE = np.array([HALFWIDTH, -HALFHEIGHT]) 
PYGAME_WINDOW_ORIGIN = np.array([HALFWIDTH, HALFHEIGHT])

# Aspect ratio that converts y to x
ASPECT_RATIO = SCREEN_WIDTH/SCREEN_HEIGHT

# FOV in y
FOVY = math.pi/2

# Distance to Near Plane -> closest value to see -> 0.1 = 10 cm
NEAR_Z = 0.1

# Distance to far plane -> furthest value to see -> 1 = 1 m
FAR_Z = 10

# NEAR_Z / (HEIGHT/2) -> y scaling factor
FY = 1/math.tan(FOVY/2)

# NEAR_Z / (WIDTH/2) -> x scaling factor
FX = (1/math.tan(FOVY/2))/ASPECT_RATIO


NEAR_PLANE_HEIGHT = 2 * math.tan(FOVY/2) * NEAR_Z
NEAR_PLANE_WIDTH = 2 * math.tan(FOVY/2) * NEAR_Z * ASPECT_RATIO

# FOV in x
FOVX = math.atan((NEAR_PLANE_WIDTH/2)/NEAR_Z) * 2

#### FRUSTUM ####
PLANES = np.empty((4,4))
PLANES[0] = np.array([-math.cos(FOVX/2), 0, math.sin(FOVX/2), 0]) # right
PLANES[1] = np.array([math.cos(FOVX/2),  0, math.sin(FOVX/2), 0]) # left
PLANES[2] = np.array([0, -math.cos(FOVY/2), math.sin(FOVY/2), 0]) # bottom
PLANES[3] = np.array([0, math.cos(FOVY/2),  math.sin(FOVY/2), 0]) # top

PLANE_NORMALS = PLANES

PLANE_POINTS = NEAR_Z * np.array([np.array([0, 0, 0, 0]), # right
                                  np.array([0, 0, 0, 0]),  # left
                                  np.array([0, 0, 0, 0]), # bottom
                                  np.array([0, 0, 0, 0])])



BACKGROUND_COLOR = (80,85,90)
SPEED = 0.01
MAX_FPS = 60
ROT_ANGLE = math.pi/360
RATE = SPEED * MAX_FPS

"""PLANES = np.array([np.array([-NEAR_Z, 0, NEAR_PLANE_WIDTH/2, 0]) , # right
                   np.array([NEAR_Z, 0, NEAR_PLANE_WIDTH/2, 0]) ,  # left
                   np.array([0, -NEAR_Z, NEAR_PLANE_HEIGHT/2, 0]) , # bottom
                   np.array([0, NEAR_Z, NEAR_PLANE_HEIGHT/2, 0])]) # top"""

# Normalizing plane normal vectors
#PLANE_NORMALS = np.array([plane/math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2) for plane in PLANES])