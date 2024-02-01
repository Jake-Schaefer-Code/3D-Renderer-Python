import numpy as np
import pygame as pg
from collections import deque
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
        
        self.lightdir = np.array([0,0,-1,0])
        
        self.generate_face_normals()

    def generate_from_obj(self, tofrust):
        polygon_vertex_indices, points, maxval, normals, faces = read_obj(f'obj_files/{self.filename}')
        self.maxval = maxval

        self.points = np.array([self.to_frustum(p) for p in points]) if tofrust else np.array([p for p in points])
        self.points = np.array([p+self.position for p in self.points])

        self.numpoints = len(points)

        self.numfaces = len(polygon_vertex_indices)
        self.normals = normals if normals != [] else [[0,0,0,0] for _ in range(self.numfaces)]
        self.faces = faces # List of dictionaries of vertices, normal, and textures for each face
        
    def generate_face_normals(self):
        for i in range(len(self.faces)):
            if self.faces[i]["vn"] == []:
                vertex_indices = self.faces[i]["v"]
                face = [self.points[j] for j in vertex_indices]
                normal = get_face_normal(face)
                nx, ny, nz, nw = normal
            else:
                nx, ny, nz, nw = np.mean(np.array([self.normals[index] for index in self.faces[i]["vn"]]), axis=0)
            
            self.faces[i]["vn"] = [nx,ny,nz,nw]

    # converts to frustum coordinates
    def to_frustum(self, point: np.ndarray):
        z = NEAR_Z*((point[2])/self.maxval) + 2* self.cam.center[2]
        x = NEAR_Z*((point[0])/self.maxval)
        y = -NEAR_Z*((point[1])/self.maxval)
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
        transform_points = Transformed_List(self.points, self.cam)
        draw_points = Draw_Point_List(transform_points, self.cam)
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
                norm_color_dot = (nx * self.lightdir[0] + ny * self.lightdir[1] + nz * self.lightdir[2])
                # Setting the last index of the face to the colorval
                face = [[transform_points[j] for j in vertex_indices], [draw_points[j] for j in vertex_indices]] + [-norm_color_dot if norm_color_dot < 0 else norm_color_dot]
                todraw.append(face)
            
        # Calls to functions that clip the mesh and z-order it
        todraw = zordermesh(todraw)
        todraw = self.clip_mesh(todraw)
        return todraw

    def draw4(self) -> None:
        draw_mesh = self.prepare_mesh()
        for face in draw_mesh:
            polygon = face[1]
            color = face[2]
            
            # Draws Polygons
            pg.draw.polygon(self.screen,(255*color,255*color,255*color), polygon, width = 0)
            # Draws edges on triangles
            pg.draw.polygon(self.screen, (20,20,20), polygon, width = 1)
    
    
# Performs transformations on all points
class Transformed_List(list):
    def __new__(cls, points, camera):
        transformed_points = [camera.transform_point(point) for point in points]
        return transformed_points

# Projects points and converts them to pygame coordinates
class Draw_Point_List(list):
    def __new__(cls, points, camera):
        pygame_points = [to_pygame(camera.perspective_projection(point)) for point in points]
        return pygame_points

