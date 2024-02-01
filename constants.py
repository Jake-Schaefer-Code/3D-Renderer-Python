import math
import numpy as np
"""Constants for 3D Renderer"""

# Desired width and height of screen
WIDTH = 1080    
HEIGHT = 720
HALFWIDTH = WIDTH/2
HALFHEIGHT = HEIGHT/2
#ASPECT_RATIO = HEIGHT/WIDTH
ASPECT_RATIO = WIDTH/HEIGHT
# Distance to far plane

#FOV (width FOV)
FOV = math.pi/2
# Distance to Near Plane
nearestApproachToPlayer = 1
NEAR_Z = nearestApproachToPlayer / (1 + math.tan(FOV/2)**2 * (ASPECT_RATIO**2 + 1))**0.5
FAR_Z = 10*NEAR_Z

# define coordinate origin to be in center of the screen, input point is in this form
SCREEN_ORIGIN = np.array([WIDTH/2, HEIGHT/2,0,1])
SCREEN_SCALE = np.array([WIDTH/2, HEIGHT/2,1,0])

# TODO add back near and far planes
PLANES = np.array([np.array([-NEAR_Z, 0, WIDTH/2, 0]) , # right
                   np.array([NEAR_Z, 0, WIDTH/2, 0]) ,  # left
                   np.array([0, -NEAR_Z, HEIGHT/2, 0]) , # bottom
                   np.array([0, NEAR_Z, HEIGHT/2, 0])]) # top


PLANES = np.array([np.array([-NEAR_Z, 0, math.tan(FOV/2)*NEAR_Z, 0]) , # right
                   np.array([NEAR_Z, 0, math.tan(FOV/2)*NEAR_Z, 0]) ,  # left
                   np.array([0, -NEAR_Z, math.tan(FOV/2)*NEAR_Z/ASPECT_RATIO, 0]) , # bottom
                   np.array([0, NEAR_Z, math.tan(FOV/2)*NEAR_Z/ASPECT_RATIO, 0])])

# Normalizing plane normal vectors
PLANE_NORMALS = np.array([plane/math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2) for plane in PLANES])

"""PLANE_DICT = {'right': np.array([-NEAR_Z, 0, WIDTH/2, 0]), 
          'left': np.array([NEAR_Z, 0, WIDTH/2, 0]), 
          'bottom': np.array([0, -NEAR_Z, HEIGHT/2, 0]),
          'top': np.array([0, NEAR_Z, HEIGHT/2, 0])}"""

PLANE_POINTS = NEAR_Z * np.array([np.array([0, 0, 0, 0]), # right
                                  np.array([0, 0, 0, 0]),  # left
                                  np.array([0, 0, 0, 0]), # bottom
                                  np.array([0, 0, 0, 0])])

BACKGROUND_COLOR = (40,40,40)
MAX_FPS = 60