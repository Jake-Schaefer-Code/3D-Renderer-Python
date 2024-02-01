import numpy as np
import math

from quaternions import *
from functions import *
from constants import *

class Camera:
    def __init__(self):
        self.pos = np.array([0,0,0,1])
        self.trans_pos = np.array([0,0,0,1])
        
        # Clipping planes (only sides, no near and far planes)
        self.planes = PLANE_NORMALS
        # FOV
        self.fovx = FOV
        # Distance to near (projection) plane
        self.nearz = NEAR_Z
        # Distance to far plane
        self.farz = FAR_Z

        # Left, Bottom, Near coordinates
        self.lbn = np.array([-math.tan(FOVX/2)*NEAR_Z, -math.tan(FOVX/2)*NEAR_Z/ASPECT_RATIO, NEAR_Z])
        # Right, Top, Far coordinates
        self.rtf = np.array([math.tan(FOVX/2)*NEAR_Z, math.tan(FOVX/2)*NEAR_Z/ASPECT_RATIO, FAR_Z])

        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = self.z_scale(np.array([0,0,0,1]))
        
        # The matrix that will be modified through transitions and rotations
        self.matrix = np.identity(4)

        # Initializing perspective projection matrix
        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l),      0,          0,       -n*(r+l)/(r-l)],
                                [    0,      2*n/(t-b),      0,       -n*(t+b)/(t-b)],
                                [    0,          0,      (n+f)/(f-n),  2*(n*f)/(n-f)],
                                [    0,          0,          1,              0      ]])

        if r == -l and t == -b:
            self.perspM[0][3] = 0
            self.perspM[1][3] = 0
    
    # Updates the camera position as if it was being moved
    def update_cam(self):
        inv_matrix = np.linalg.inv(self.matrix)
        self.trans_pos = inv_matrix @ self.pos

    # Applying transformation matrix
    def transform_point(self, point) -> np.ndarray:
        return self.matrix @ point
    
    # Projecting point onto projection plane
    def perspective_projection(self, point: np.ndarray) -> np.ndarray:
        persp_point = self.perspM @ point
        return persp_point / (persp_point[3] if persp_point[3] != 0 else 1e-6)

    # Camera movement
    def move_cam(self, trans_vector: np.ndarray) -> None:
        self.matrix = translate(self.matrix, trans_vector)

    # Camera Rotation
    def rotate_cam(self, axis: np.ndarray, angle: float) -> None:
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

