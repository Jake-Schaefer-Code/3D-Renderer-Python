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
        # Clipping planes (only sides, no near and far planes)
        self.clipping_planes = PLANE_NORMALS
        # FOV
        self.fovy = FOVY
        # Distance to near (projection) plane
        self.nearz = NEAR_Z
        # Distance to far plane
        self.farz = FAR_Z

        # Left, Bottom, Near coordinates
        self.lbn = np.array([-NEAR_PLANE_WIDTH/2, -NEAR_PLANE_HEIGHT/2, NEAR_Z])
        # Right, Top, Far coordinates
        self.rtf = np.array([ NEAR_PLANE_WIDTH/2, NEAR_PLANE_HEIGHT/2, FAR_Z])


        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = self.z_scale(np.array([0,0,0,1]))
        
        # The matrix that will be modified through transitions and rotations
        self.matrix = np.identity(4)
        self.modelview_matrix = np.identity(4)

        # Initializing perspective projection matrix
        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l),      0,          0,       -n*(r+l)/(r-l)],
                                [    0,      2*n/(t-b),      0,       -n*(t+b)/(t-b)],
                                [    0,          0,      -(n+f)/(n-f), 2*(n*f)/(n-f)],
                                [    0,          0,          1,              0      ]], dtype='float32')
        

        self.projM = create_projection_matrix(self.lbn, self.rtf)
        self.projMT = self.projM.T
        

        if r == -l and t == -b:
            self.projM[0][2] = 0
            self.projM[1][2] = 0


        # Movement stuff
        self.rate = SPEED * MAX_FPS
        self.move_z = np.array([0,0,self.rate,0])
        self.move_y = np.array([0,self.rate,0,0])
        self.move_x = np.array([self.rate,0,0,0])
    
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

    def transform_point(self, point:np.ndarray) -> np.ndarray:
        """
        Applies the transformation matrix on a single point

        Parameters:
        ----------------
        point : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        return self.matrix @ point
    
    # Applying transformation matrix over an array
    def transform_point_array(self, transposed_points:np.ndarray) -> np.ndarray:
        """
        Transforms an array of transposed points using the camera matrix

        Parameters:
        ----------------
        transposed_points : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        return (self.matrix @ transposed_points).T
    
    def perspective_projection(self, points: np.ndarray) -> np.ndarray:
        """
        Projects a single, or an array of points onto the projection plane

        Parameters:
        ----------------
        points : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        """if points.ndim == 1:
            persp_point = self.projM @ points
            return persp_point / (persp_point[3] if persp_point[3] != 0 else 1)
        elif points.ndim == 2:
            projected_points = points @ self.projM 
            w = projected_points[:, 3]
            return projected_points / np.where(w == 0, 1, w)[:, None]"""
        
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
    def rotate_cam(self, axis: np.ndarray, angle: float) -> None:
        """
        What this does...

        Parameters:
        ----------------
        axis : np.ndarray

        angle : float

        """
        self.matrix = q_mat_rot(self.matrix, axis, angle)
            
    # converts to canonical viewing space
    def z_scale(self, point: np.ndarray) -> np.ndarray:
        """
        What this does...

        Parameters:
        ----------------
        point : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        n = self.nearz
        f = self.farz
        c1 = 2*f*n/(n-f)
        c2 = (f+n)/(f-n)
        z = c1/(point[2]-c2)
        x = point[0]
        y = -point[1]
        return np.array([x,y,z,point[3]])