from functions import *
from constants import *
from camera import *
from objreader import *

class Object:
    def __init__(self, camera: Camera, screen: pg.display, filename: str, position: np.ndarray, tofrust: bool = True):
        self.cam = camera
        self.screen = screen
        self.position = position
        self.filename = filename
        self.generate_from_obj(tofrust)

        self.planes = PLANE_NORMALS
        self.plane_points = PLANE_POINTS
        self.generate_face_normals()
        self.oldtime = 0
        self.newtime = 0

    def generate_from_obj(self, tofrust):
        polygon_vertex_indices, points, maxval, normals, faces = read_obj(f'obj_files/{self.filename}')
        self.maxval = maxval
        self.polygon_indices = polygon_vertex_indices

        self.points = np.array([self.to_frustum(p) for p in points]) if tofrust else np.array([p for p in points])
        self.points = np.array([p+self.position for p in self.points], dtype='float32')
        self.transposed_points = self.points.T

        self.numpoints = len(points)
        self.numfaces = len(polygon_vertex_indices)
        self.normals = normals if normals != [] else [[0,0,0,0] for _ in range(self.numfaces)]
        self.faces = faces # List of dictionaries of vertices, normal, and textures for each face
        
    def generate_face_normals(self):
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

    # converts to frustum coordinates
    def to_frustum(self, point: np.ndarray):
        z = NEAR_Z * ((point[2])/self.maxval) + 2 * self.cam.center[2]
        x = NEAR_Z * ((point[0])/self.maxval)
        y = -NEAR_Z * ((point[1])/self.maxval)
        return np.array([x,y,z,point[3]])

    def clip_mesh(self, mesh: list[list]):
        triangle_queue = deque(mesh) 
        for plane in self.planes:
            for _ in range(len(triangle_queue)):
                polygon = triangle_queue.popleft()
                if len(polygon[0]) == 3: # Is triangle: 3 vertices and color
                    new_triangles = clip_triangle(polygon, plane)
                    triangle_queue.extend(new_triangles)
                else:
                    triangle_queue.append(polygon)
        return list(triangle_queue)

    def prepare_mesh(self) -> list:
        #start = time.time()
        #transform_points = Transformed_List(self.points, self.cam)
        #start = time.time()
        transform_points = (self.cam.matrix @ self.transposed_points).T
        #tp = (self.cam.matrix @ self.transposed_points).T

        #start = time.time()
        #draw_points = Draw_Point_List(transform_points, self.cam)
        draw_points = vectorized_to_pygame2(self.cam.project_points(transform_points))

        #start = time.time()

        #dp = self.cam.project_points(tp)
        
        #tpg = vectorized_to_pygame(dp)

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
        todraw = self.clip_mesh(todraw)
        return todraw

    def draw(self) -> None:
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
        pg_mesh = vectorized_to_pygame(proj_mesh)

        for i in range(len(proj_mesh)):
            color = colors[i][0]
            pg.draw.polygon(self.screen,(255*color,255*color,255*color), pg_mesh[i], width = 0)
            pg.draw.polygon(self.screen, (255,255,255), pg_mesh[i], width = 1)
        
    ### VECTORIZED ###
    def _backface_culling(self):
        """center_mesh = np.array([[np.array([0,0,0,1]), np.array([0,0,0,1]), np.array([0,0,0,1]), np.array([1,1,1,1])]])
        transformed_mesh = np.zeros((center_mesh.shape[0], 4, 4))
        for i in range(center_mesh.shape[0]):
            for j in range(3):
                transpoint = self.cam.matrix @ center_mesh[i, j]
                transformed_mesh[i, j, :] = transpoint

        transformed_mesh[:,3,:] = center_mesh[:,3,:]
        center = np.array([np.array([0,0,0,1]), np.array([0,0,0,1]), np.array([0,0,0,1]), np.array([1,1,1,1])])
        tcenter = (self.cam.matrix @ center.T).T[0]
        pcenter = self.cam.perspective_projection(tcenter)
        vpcenter, c = self.cam.project_mesh(transformed_mesh)"""
        
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
        norm_color_dots = np.abs(np.dot(self.normals[in_sight], self.cam.lightdir))


        #norm_color_dots = np.dot(transformnormals[in_sight], self.cam.lightdir) / 2
        
        #color_vals = norm_color_dots ** 2
        #color_vals = np.where(norm_color_dots < 0, -norm_color_dots, norm_color_dots)
        
        color_and_polygons[:, -1, :] = norm_color_dots[:, None]
        #color_and_polygons[:, -1, :] = np.abs(np.dot(self.normals[in_sight], self.cam.lightdir))[:, None]
        return color_and_polygons

    def _clip_mesh_vectorized(self, mesh: np.ndarray) -> np.ndarray:
        triangle_queue = deque(mesh) 
        for plane in self.cam.clipping_planes:
            for _ in range(len(triangle_queue)):
                polygon = triangle_queue.popleft()
                if len(polygon) == 4: # Is triangle: 3 vertices and color
                    new_triangles = vectorized_clip_triangle(polygon, plane)

                    triangle_queue.extend(new_triangles)
                else:
                    triangle_queue.append(polygon)

        if len(list(triangle_queue)) == 0:
            return np.zeros((1, mesh.shape[1], mesh.shape[2]))
        else:
            return np.array(list(triangle_queue))
        
    def prepare_mesh_vectorized(self) -> np.ndarray:
        """
        Internal Function
        """
        self.cam.update_cam()
        #culled_polygons, projected_polygons = self.backface_culling()
        culled_polygons = self._backface_culling()
        #reorder = vectorized_zordermesh2(culled_polygons[:,:-1,:])
        #projected_polygons = projected_polygons[reorder]
        #todraw = self.clip_mesh_vectorized2(projected_polygons)
        todraw = vectorized_zordermesh(culled_polygons)
        todraw = self._clip_mesh_vectorized(todraw)
        return todraw
    
    """def prepare_mesh_vectorized2(self) -> np.ndarray:
        '''
        Internal Function
        '''

        transform_points = (self.cam.matrix @ self.transposed_points).T
        projected_points = self.cam.project_points(transform_points)
        draw_points = vectorized_to_pygame(projected_points)
        self.cam.update_cam()
        in_sight = ((self.points[self.polygon_indices[:, 1]] - self.cam.trans_pos) * self.normals).sum(axis=1) < 0
        visible_polygons = transform_points[self.polygon_indices]
        projected_polygons = draw_points[self.polygon_indices]
        norm_color_dots = np.dot(self.normals[in_sight], self.cam.lightdir)

        z_order_indices = vectorized_zordermesh2()
        visible_polygons = visible_polygons[z_order_indices]
        projected_polygons = projected_polygons[z_order_indices]
        color_vals = np.where(norm_color_dots < 0, -norm_color_dots, norm_color_dots)[z_order_indices]

        todraw = self.clip_mesh(np.array([visible_polygons, projected_polygons, color_vals]))
        #todraw = self.clip_mesh(todraw)
        return todraw
    
    def clip_mesh_vectorized2(self, mesh: np.ndarray) -> np.ndarray:
        triangle_queue = deque(mesh)
        for i in range(len(self.planes)):
            plane = PROJECTED_PLANES[i]
            plane_point = PROJECTED_PLANE_POINTS[i]
            for _ in range(len(triangle_queue)):
                polygon = triangle_queue.popleft()
                if len(polygon) == 3: # Is triangle: 3 vertices and color
                    new_triangles = vectorized_clip_triangle2(polygon, plane, plane_point)

                    triangle_queue.extend(new_triangles)
                else:
                    triangle_queue.append(polygon)

        if len(list(triangle_queue)) == 0:
            return np.zeros((1, mesh.shape[1], mesh.shape[2]))
        else:
            return np.array(list(triangle_queue))"""




    
    
# Performs transformations on all points
class Transformed_List(list):
    def __new__(cls, points, camera: Camera):
        return [camera.transform_point(point) for point in points]

# Projects points and converts them to pygame coordinates
class Draw_Point_List(list):
    def __new__(cls, points, camera: Camera):
        return [to_pygame(camera.perspective_projection(point)) for point in points]

