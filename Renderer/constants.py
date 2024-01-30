import math
import numpy as np
# Constants for 3D Renderer
WIDTH = 1080    
HEIGHT = 720
HALFWIDTH = WIDTH/2
HALFHEIGHT = HEIGHT/2
ASPECT_RATIO = HEIGHT/WIDTH

FAR_Z = 3600
FOVX = math.pi/2
FOVY = 2 * math.atan(ASPECT_RATIO * math.tan(FOVX/2))
NEAR_Z = WIDTH/(2*math.tan(FOVX/2))


# define coordinate origin to be in center of the screen, input point is in this form
SCREEN_ORIGIN = np.array([WIDTH/2, HEIGHT/2,0,1])
SCREEN_SCALE = np.array([WIDTH/2, HEIGHT/2,1,0])
PLANES = np.array([np.array([-math.sin(FOVX/2), 0, math.cos(FOVX/2), 0]) , # right
          np.array([math.sin(FOVX/2), 0, math.cos(FOVX/2), 0]) ,  # left
          np.array([0, -math.cos(FOVY/2), math.sin(FOVY/2), 0]) , # bottom (because y coords are inverted on screen)
          np.array([0, math.cos(FOVY/2), math.sin(FOVY/2), 0]) ,  # top
          np.array([0, 0, 1, 0]),   # near
          np.array([0, 0, -1, 0])])  # far

PLANES = np.array([np.array([-NEAR_Z, 0, WIDTH/2, 0]) , # right
          np.array([NEAR_Z, 0, WIDTH/2, 0]) ,  # left
          np.array([0, -NEAR_Z, HEIGHT/2, 0]) , # bottom (because y coords are inverted on screen)
          np.array([0, NEAR_Z, HEIGHT/2, 0])])
          
          #,np.array([0, 0, 1, 0])] # top

"""PLANES = {'right': np.array([-NEAR_Z, 0, WIDTH/2, 0]), 
          'left': np.array([NEAR_Z, 0, WIDTH/2, 0]), 
          'bottom': np.array([0, -NEAR_Z, HEIGHT/2, 0]),
          'top': np.array([0, NEAR_Z, HEIGHT/2, 0])}"""

# TODO add back near and far planes

PLANES = np.array([plane/math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2) for plane in PLANES])




PLANE_POINTS = NEAR_Z * np.array([np.array([0, 0, 0, 0]), # right
                         np.array([0, 0, 0, 0]),  # left
                         np.array([0, 0, 0, 0]), # bottom
                         np.array([0, 0, 0, 0]),
                         ]) 

BACKGROUND_COLOR = (40,40,40)