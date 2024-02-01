import math
import numpy as np
"""Constants for 3D Renderer"""

# Desired width and height of screen
WIDTH = 1080    
HEIGHT = 720
HALFWIDTH = WIDTH/2
HALFHEIGHT = HEIGHT/2
ASPECT_RATIO = WIDTH/HEIGHT


# FOV (width FOV)
FOV = math.pi/2
# Could be replaced with hitbox code later on
NEAREST_VAL = 1
# Distance to Near Plane
NEAR_Z = NEAREST_VAL / (1 + math.tan(FOV/2)**2 * (ASPECT_RATIO**2 + 1))**0.5
# Distance to far plane
FAR_Z = 10*NEAR_Z

NEAR_PLANE_WIDTH = 2 * math.tan(FOV/2)*NEAR_Z
NEAR_PLANE_HEIGHT = 2 * math.tan(FOV/2)*NEAR_Z/ASPECT_RATIO

# define coordinate origin to be in center of the screen, input point is in this form
SCREEN_ORIGIN = np.array([WIDTH/2, HEIGHT/2,0,1])
SCREEN_SCALE = np.array([WIDTH/2, HEIGHT/2,1,0])

PLANES = np.array([np.array([-NEAR_Z, 0, NEAR_PLANE_WIDTH/2, 0]) , # right
                   np.array([NEAR_Z, 0, NEAR_PLANE_WIDTH/2, 0]) ,  # left
                   np.array([0, -NEAR_Z, NEAR_PLANE_HEIGHT/2, 0]) , # bottom
                   np.array([0, NEAR_Z, NEAR_PLANE_HEIGHT/2, 0])]) # top

# Normalizing plane normal vectors
PLANE_NORMALS = np.array([plane/math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2) for plane in PLANES])

PLANE_POINTS = NEAR_Z * np.array([np.array([0, 0, 0, 0]), # right
                                  np.array([0, 0, 0, 0]),  # left
                                  np.array([0, 0, 0, 0]), # bottom
                                  np.array([0, 0, 0, 0])])

BACKGROUND_COLOR = (40,40,40)
SPEED = 0.025
MAX_FPS = 60