import math
import numpy as np
# Constants for 3D Renderer
WIDTH = 720
HEIGHT = 720
ASPECT_RATIO = HEIGHT/WIDTH
FOVX = math.pi/2    
FOVY = 2 * math.atan(ASPECT_RATIO * math.tan(FOVX/2))

# define coordinate origin to be in center of the screen, input point is in this form
SCREEN_ORIGIN = np.array([WIDTH/2, HEIGHT/2,0,1])
SCREEN_SCALE = np.array([WIDTH/2, HEIGHT/2,1,0])
PLANES = [np.array([-math.sin(FOVX/2), 0, math.cos(FOVX/2), 0]), # right
          np.array([math.sin(FOVX/2), 0, math.cos(FOVX/2), 0]),  # left
          np.array([0, -math.sin(FOVY/2), math.cos(FOVY/2), 0]), # top
          np.array([0, math.sin(FOVY/2), math.cos(FOVY/2), 0]),  # bottom
          np.array([0, 0, 1, 0]),   # near
          np.array([0, 0, -1, 0])]  # far

NEAR_Z = 360
FAR_Z = 3600

PLANE_POINTS = [np.array([NEAR_Z*math.tan(FOVX/2), 0, NEAR_Z, 0]), # right
                np.array([-NEAR_Z*math.tan(FOVX/2), 0, NEAR_Z, 0]),  # left
                np.array([0, NEAR_Z*math.tan(FOVY/2), NEAR_Z, 0]), # top
                np.array([0, -NEAR_Z*math.tan(FOVY/2), NEAR_Z, 0]),  # bottom
                np.array([0, 0, 10, 0]), # near
                np.array([0, 0, 3600, 0])]  # far
center = np.array([0,0,654.545])


"""l,b,n = self.nearz * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), -1.0])
        r,t,f = self.nearz * np.array([math.tan(FOVX/2), math.tan(FOVY/2), -self.farz/self.nearz])
        self.perspM = np.array([[2*n/(r-l),0,0,-n*(r+l)/(r-l)],
                        [0,2*n/(t-b),0,-n*(t+b)/(t-b)],
                        [0,0,-(n+f)/(f-n),2*n*f/(n-f)],
                        [0,0,-1,0]])"""