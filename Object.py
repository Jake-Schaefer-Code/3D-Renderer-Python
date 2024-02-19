from functions import *
from constants import *
from camera import *
from objreader import *

class Object:
    def __init__(self, camera: Camera, screen: pg.display, filename: str, position: np.ndarray, tofrust: bool = True):
        """
        What this does...

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
        self.filename = filename
        self.generate_from_obj(tofrust)

        self.planes = PLANE_NORMALS
        self.plane_points = PLANE_POINTS
        self._generate_face_normals()
        self.oldtime = 0
        self.newtime = 0

    def generate_from_obj(self, tofrust: bool):
        """
        What this does...

        Parameters:
        ----------------
        tofrust : bool
        """
        points, normals, component_array = read_obj(f'obj_files/{self.filename}')
        #polygon_vertex_indices, points, normals, faces, component_array = read_obj(f'obj_files/{self.filename} ')
        self.component_array = component_array
        self.maxval = np.max(np.abs(points))
        self.polygon_indices = component_array['v']
        self.points = to_frustum(points, self.maxval, self.cam.center) if tofrust else points
        #self.points = np.array([self.to_frustum(p) for p in points]) if tofrust else np.array([p for p in points])
        self.points = np.array([p+self.position for p in self.points], dtype='float32')
        self.transposed_points = self.points.T
        self.numpoints = len(points)
        self.numfaces = len(component_array['v'])
        self.normals = np.array(normals) if normals != [] else np.zeros((self.numfaces, 4))
        
    def _generate_face_normals(self) -> None:
        """
        What this does...

        """
        normals = np.zeros((self.numfaces, 4))
        none_indices = np.array([-1 in normal for normal in self.component_array['vn']])
        normals[none_indices] = get_face_normal_vectorized(self.points[self.component_array['v'][none_indices]])
        not_none = ~none_indices
        indices = self.component_array['vn'][not_none]
        normals[not_none] = np.mean(self.normals[indices], axis=1)
        self.normals = normals
        self.transposed_normals = self.normals.T
        
        """
        normals = []
        for i in range(len(self.faces)):
            if self.faces[i]["vn"] == []:
                vertex_indices = self.faces[i]["v"]
                face = [self.points[j] for j in vertex_indices]
                normal = get_face_normal(face)
                nx, ny, nz, nw = normal
            else:
                nx, ny, nz, nw = np.mean(np.array([self.normals[index] for index in self.faces[i]["vn"]]), axis=0)
            
            self.faces[i]["vn"] = [nx,ny,nz,nw]
            normals.append([nx,ny,nz,nw])
        self.normals = np.array(normals, dtype='float32')
        self.transposed_normals = self.normals.T
        """

    def prepare_mesh(self) -> list:
        """
        What this does...

        Returns:
        ----------------
        list
        """
        #start = time.time()
        #transform_points = Transformed_List(self.points, self.cam)
        #start = time.time()
        transform_points = (self.cam.matrix @ self.transposed_points).T
        #tp = (self.cam.matrix @ self.transposed_points).T

        #start = time.time()
        #draw_points = Draw_Point_List(transform_points, self.cam)
        draw_points = to_pygame(self.cam.perspective_projection(transform_points))

        #start = time.time()

        #dp = self.cam.perspective_projection(tp)
        
        #tpg = to_pygame(dp)

        #transform_points = [self.cam.transform_point(point) for point in self.points]
        #draw_points = [to_pygame(self.cam.perspective_projection(point)) for point in transform_points]
        self.cam.update_cam()

        todraw = []
        for face_values in self.faces:
            vertex_indices = face_values['v']
            normal = face_values['vn']
            nx = normal[0]
            ny = normal[1]
            nz = normal[2]

            plane_point = self.points[vertex_indices[1]]
            cam_pos = self.cam.trans_pos
            # Back-face culling: only draw if face is facing camera i.e. if normal is facing in negative direction
            # This is the normal dotted with the view vector pointing from the polygon's surface to the camera's position
            coincide = ((plane_point[0] - cam_pos[0]) * nx + (plane_point[1] - cam_pos[1]) * ny + (plane_point[2] - cam_pos[2]) * nz)
            if coincide < 0:
                lightdir = self.cam.lightdir
                norm_color_dot = (nx * lightdir[0] + ny * lightdir[1] + nz * lightdir[2])
                # Setting the last index of the face to the colorval
                face = [[transform_points[j] for j in vertex_indices], [draw_points[j] for j in vertex_indices]] + [-norm_color_dot if norm_color_dot < 0 else norm_color_dot]
                todraw.append(face)
        
       
        # Calls to functions that clip the mesh and z-order it
        todraw = zordermesh(todraw)
        #todraw = self.clip_mesh(todraw)
        return todraw

    def draw(self) -> None:
        """
        What this does...
        """
        """draw_mesh = self.prepare_mesh()
        for face in draw_mesh:
            polygon = face[1]
            color = face[2]
            # Draws Polygons
            pg.draw.polygon(self.screen,(255*color,255*color,255*color), polygon, width = 0)
            # Draws edges on triangles
            pg.draw.polygon(self.screen, (20,20,20), polygon, width = 1)"""

        vec_mesh = self.prepare_mesh_vectorized()
        proj_mesh, colors = self.cam.project_mesh(vec_mesh)
        pg_mesh = to_pygame(proj_mesh)

        for i in range(len(proj_mesh)):
            color = colors[i][0]
            pg.draw.polygon(self.screen,(255*color,255*color,255*color), pg_mesh[i], width = 0)
            pg.draw.polygon(self.screen, (255,255,255), pg_mesh[i], width = 1)
        
    def _backface_culling(self) -> np.ndarray:
        """
        What this does...

        Returns:
        ----------------
        np.ndarray
        """
        #transformpoints = ((self.cam.matrix @ self.transposed_points).T)

        #tp = self.cam.matrix @ self.transposed_points
        #transformpoints = tp.T
        #projectedpoints = (self.cam.perspM @ tp).T
        #projected_polygons = projectedpoints[self.polygon_indices]
        #transformnormals = (self.cam.matrix @ self.transposed_normals).T
        
        in_sight = ((self.points[self.polygon_indices[:, 1]] - self.cam.trans_pos) * self.normals).sum(axis=1) < 0
        #in_sight = (transformnormals * polygons[:, 1]).sum(axis=1) < 0
        
        #polygons = transformpoints[self.polygon_indices]
        #visible_polygons = polygons[in_sight]
        
        visible_polygons = ((self.cam.matrix @ self.transposed_points).T)[self.polygon_indices][in_sight]

        color_and_polygons = np.zeros((visible_polygons.shape[0], visible_polygons.shape[1] + 1, visible_polygons.shape[2]))
        color_and_polygons[:,:-1,:] = visible_polygons
        norm_color_dots = np.abs(np.dot(self.normals[in_sight], self.cam.trans_lightdir))
        norm_color_dots = norm_color_dots / np.max(norm_color_dots)

        #norm_color_dots = np.dot(transformnormals[in_sight], self.cam.lightdir) / 2
        
        #color_vals = norm_color_dots ** 2
        #color_vals = np.where(norm_color_dots < 0, -norm_color_dots, norm_color_dots)
        
        color_and_polygons[:, -1, :] = norm_color_dots[:, None]
        #color_and_polygons[:, -1, :] = np.abs(np.dot(self.normals[in_sight], self.cam.lightdir))[:, None]
        return color_and_polygons

    def _clip_mesh_vectorized(self, mesh: np.ndarray) -> np.ndarray:
        """
        What this does...

        Parameters:
        ----------------
        mesh : np.ndarray

        ----------------
        Returns:
        ----------------
        np.ndarray
        """
        #triangle_queue = deque(mesh) 
        nullshape = (1, mesh.shape[1], mesh.shape[2])
        for plane in self.cam.clipping_planes:
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
        
    def prepare_mesh_vectorized(self) -> np.ndarray:
        """
        What this does...

        Returns:
        ----------------
        np.ndarray
        """
        self.cam.update_cam()
        culled_polygons = self._backface_culling()
        todraw = zordermesh(culled_polygons)
        todraw = self._clip_mesh_vectorized(todraw)
        return todraw
    


# Performs transformations on all points
class Transformed_List(list):
    def __new__(cls, points, camera: Camera):
        return [camera.transform_point(point) for point in points]

# Projects points and converts them to pygame coordinates
class Draw_Point_List(list):
    def __new__(cls, points, camera: Camera):
        return [to_pygame(camera.perspective_projection(point)) for point in points]

