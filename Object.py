from functions import *
from constants import *
from camera import *
from objreader import *

class Object:
    def __init__(self, camera: Camera, screen: pg.display, fileName: str, position: np.ndarray, tofrust: bool = True):
        """
        Parameters:
        ----------------
        camera : Camera

        screen : pg.display

        filename : str

        position : np.ndarray

        tofrust : bool

        """
        self.cam = camera
        self.screen = screen
        self.position = position
        self.fileName = fileName
        self.objFile = OBJFile(f'{os.path.dirname(__file__)}/obj_files/{fileName}')
        self._generate_from_obj(tofrust)
        self._generate_face_normals()

        self.planes = PLANE_NORMALS
        self.plane_points = PLANE_POINTS
        self.mono_color = np.array([255,255,255])

        

    def _generate_from_obj(self, tofrust: bool) -> None:
        """
        Reads the obj file and gets attributes

        Parameters:
        ----------------
        tofrust : bool
        """
        self.objFile.read_obj()
        
        points = self.objFile.vertices
        self.points = to_frustum(points + self.position, self.objFile.maxval, 0.5*self.cam.center) if tofrust else points + self.position
        self.transposed_points = self.points.T
        self.numpoints = len(self.points)

        self.normals = self.objFile.normals

        self.component_array = self.objFile.component_array
        self.polygon_indices = self.component_array['v']
        self.numfaces = len(self.component_array['v'])
    
        
    def _generate_face_normals(self) -> None:
        """
        Generates the normals of each polygon in the object

        """
        normals = np.zeros((self.numfaces, 4))
        none_indices = np.array([np.any(normal <= -1) for normal in self.component_array['vn']])
        normals[none_indices] = get_face_normal_vectorized(self.points[self.component_array['v'][none_indices]])
        not_none = ~none_indices
        indices = self.component_array['vn'][not_none]
        if np.any(not_none):
            normals[not_none] = np.mean(self.normals[indices], axis=1)
        self.normals = normals
        self.transposed_normals = self.normals.T

    def draw(self) -> None:
        """
        Puts the object through the rendering pipeline and draws it to the screen

        """
        
        vec_mesh = self.mesh_prep_pipeline()
        proj_mesh, colors = self.cam.project_mesh(vec_mesh)
        pg_mesh = to_pygame(proj_mesh)
        
        for i in range(len(proj_mesh)):
            color = self.mono_color * colors[i][0]
            #(255*shading,255*shading,255*shading)
            pg.draw.polygon(self.screen, color, pg_mesh[i], width = 0)
            #pg.draw.polygon(self.screen, (255,255,255), pg_mesh[i], width = 1)
        
    def _backface_culling(self) -> np.ndarray:
        """
        Keeps only the polygons which are facing towards the camera (normals that are anti-parallel thru perpendicular to the camera veiw vector)

        Returns:
        ----------------
        color_and_polygons : np.ndarray
            An array of only the polygons facing the camera
        """
        #transformpoints = ((self.cam.matrix @ self.transposed_points).T)
        #tp = self.cam.matrix @ self.transposed_points
        #transformpoints = tp.T
        #projectedpoints = (self.cam.perspM @ tp).T
        #projected_polygons = projectedpoints[self.polygon_indices]
        #transformnormals = (self.cam.matrix @ self.transposed_normals).T
        #in_sight = (transformnormals * polygons[:, 1]).sum(axis=1) < 0
        #polygons = transformpoints[self.polygon_indices]
        #visible_polygons = polygons[in_sight]

        
        in_sight = ((self.points[self.polygon_indices[:, 1]] - self.cam.trans_pos) * self.normals).sum(axis=1) < 0
        
        visible_polygons = self.cam.transform_point_array(self.points)[self.polygon_indices][in_sight]

        color_and_polygons = np.zeros((visible_polygons.shape[0], visible_polygons.shape[1] + 1, visible_polygons.shape[2]))
        color_and_polygons[:,:-1,:] = visible_polygons

        norm_color_dots = self._get_normal_shading(self.normals[in_sight], self.cam.trans_lightdir)
        """norm_color_dots = np.abs(np.dot(self.normals[in_sight], self.cam.trans_lightdir))
        norm_color_dots = norm_color_dots / np.max(norm_color_dots)"""
        color_and_polygons[:, -1, :] = norm_color_dots[:, None]
        return color_and_polygons

    def _get_normal_shading(self, normals: np.ndarray, light_vector: np.ndarray) -> np.ndarray:
        """
        Gets shading factor of polygons based on their normals and a light vector

        Parameters:
        ----------------
        normals : np.ndarray

        light_vector : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        norm_color_dots = np.abs(np.dot(normals, light_vector))
        return norm_color_dots / np.max(norm_color_dots)


    def _clip_mesh(self, mesh: np.ndarray) -> np.ndarray:
        """
        Clips the input mesh against the frustum clipping planes

        Parameters:
        ----------------
        mesh : np.ndarray

        ----------------
        Returns:
        ----------------
        mesh : np.ndarray
        """
        #triangle_queue = deque(mesh) 
        nullshape = (1, mesh.shape[1], mesh.shape[2])
        for plane in PLANE_NORMALS:
            new_clipped_polygons = []
            for polygon in mesh:
                #if len(polygon) == 4: # Is triangle: 3 vertices and color
                new_triangles = clip_triangle(polygon, plane)
                new_clipped_polygons.extend(new_triangles)
            mesh = np.array(new_clipped_polygons)
        if len(mesh) == 0:
            return np.zeros(nullshape)
        else:
            return mesh
        
    def mesh_prep_pipeline(self) -> np.ndarray:
        """
        What this does...

        Returns:
        ----------------
        np.ndarray
        """
        self.cam.update_cam()
        culled_polygons = self._backface_culling()
        todraw = zordermesh(culled_polygons)
        todraw = self._clip_mesh(todraw)
        return todraw
    


# Performs transformations on all points
class Transformed_List(list):
    def __new__(cls, points, camera: Camera):
        return [camera.transform_point(point) for point in points]

# Projects points and converts them to pygame coordinates
class Draw_Point_List(list):
    def __new__(cls, points, camera: Camera):
        return [to_pygame(camera.perspective_projection(point)) for point in points]

