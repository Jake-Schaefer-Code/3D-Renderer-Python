from quaternions import *
from functions import *
from constants import *
from matrices import *

class Camera:
    def __init__(self):
        self.pos = np.array([0,0,0,1])
        self.trans_pos = np.array([0,0,0,1])
        self.lightdir = np.array([0,0,-1,0])
        self.trans_lightdir = np.array([0,0,-1,0])
        self.clock = pg.time.Clock()

        # Left, Bottom, Near coordinates
        self.lbn = np.array([-NEAR_PLANE_WIDTH/2, -NEAR_PLANE_HEIGHT/2, NEAR_Z])
        # Right, Top, Far coordinates
        self.rtf = np.array([ NEAR_PLANE_WIDTH/2, NEAR_PLANE_HEIGHT/2, FAR_Z])

        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = z_scale(self.pos)
        
        # The matrix that will be modified through transitions and rotations
        self.matrix = np.identity(4)
        self.modelview_matrix = np.identity(4)
        
        # Initializing perspective projection matrix
        self.projM = create_projection_matrix(self.lbn, self.rtf)
        self.projMT = self.projM.T

        # Movement stuff
        self.velocity = RATE
        self.move_z = np.array([0,0,1,0]) * self.velocity
        self.move_y = np.array([0,1,0,0]) * self.velocity
        self.move_x = np.array([1,0,0,0]) * self.velocity
        self.held = False
    
    # Updates the camera position as if it was being moved
    def update_cam(self):
        self.modelview_matrix = np.linalg.inv(self.matrix)
        self.trans_pos = self.modelview_matrix @ self.pos
        self.trans_lightdir = self.modelview_matrix @ self.lightdir

    def check_movement(self):
        # if want simultaneous movement change these all to ifs
        fps = self.clock.get_fps()
        # CONTROLS
        keys = pg.key.get_pressed()
        if fps == 0:
            fps = 30
        if keys[pg.K_w]:
            self.move_cam(self.move_z/fps)
        elif keys[pg.K_s]:
            self.move_cam(-self.move_z/fps)
        if keys[pg.K_d]:
            self.move_cam(self.move_x/fps)
        elif keys[pg.K_a]:
            self.move_cam(-self.move_x/fps)
        if keys[pg.K_SPACE]:
            self.move_cam(self.move_y/fps)
        elif keys[pg.K_LSHIFT]:
            self.move_cam(-self.move_y/fps)      
    
    # Applying transformation matrix over an array
    def transform_point_array(self, points: np.ndarray) -> np.ndarray:
        """
        Applies the camera transformation matrix on a single point or an array of points

        Parameters:
        ----------------
        points : np.ndarray
            An array of points to be transformed
        ----------------
        Returns:
        ----------------
        np.ndarray
            An array of transformed points
        """
        #tp = (self.matrix @ transposed_points).T
        return points @ self.matrix.T
    
    def perspective_projection(self, points: np.ndarray) -> np.ndarray:
        """
        Projects a single, or an array of points onto the projection plane

        Parameters:
        ----------------
        points : np.ndarray
            An array of points to be projected
        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        
        projected_points = points @ self.projMT
        w = projected_points[..., 3]
        return projected_points / np.where(w == 0, 1, w)[..., None]
        
    def project_mesh(self, mesh:np.ndarray) -> np.ndarray:
        """
        Projects a mesh of polygons onto the projection plane

        Parameters:
        ----------------
        mesh : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        projected_mesh = np.einsum('ij,nkj->nki', self.projM, mesh[:, :3, :])
        colors = mesh[:,-1,:]
        w = projected_mesh[:, :, 3]
        
        normalized = projected_mesh / np.where(w == 0, 1, w)[:, :, None]
        return normalized, colors

    # Camera movement
    def move_cam(self, trans_vector: np.ndarray) -> None:
        """
        What this does...

        Parameters:
        ----------------
        trans_vector : np.ndarray

        """
        self.matrix = translate_matrix(self.matrix, trans_vector)

    # Camera Rotation
    def rotate(self, axis: np.ndarray, angle: float) -> None:
        """
        Rotates the camera about an axis by an angle (in radians)

        Parameters:
        ----------------
        axis : np.ndarray

        angle : float

        """
        self.matrix = q_mat_rot(self.matrix, axis, angle)



"""if points.ndim == 1:
    persp_point = self.projM @ points
    return persp_point / (persp_point[3] if persp_point[3] != 0 else 1)
elif points.ndim == 2:
    projected_points = points @ self.projM 
    w = projected_points[:, 3]
    return projected_points / np.where(w == 0, 1, w)[:, None]"""