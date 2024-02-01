import numpy as np
import pygame as pg
from collections import deque
from functions import *
from constants import *
from camera import *

class Object:
    def __init__(self, points: np.ndarray, position: np.ndarray, indices: np.ndarray, camera: Camera, 
                 screen: pg.display, normals: np.ndarray, all_indices: np.ndarray, maxval: float=1.0, tofrust: bool = True):
        self.cam = camera
        self.screen = screen
        self.maxval = maxval
        
        self.planes = PLANE_NORMALS
        self.plane_points = PLANE_POINTS

        self.points = np.array([self.to_frustum(p) for p in points]) if tofrust else np.array([p for p in points])
        # TODO move this elsewhere?
        self.points = np.array([p+position for p in self.points])

        self.numpoints = len(points)
        self.polygons = indices
        self.numfaces = len(self.polygons)
        self.lightdir = np.array([0,0,-1,0])
        
        self.transformpoints = points
        self.normals = normals if normals != [] else [[0,0,0,0] for _ in range(self.numfaces)]
        
        
        self.all_indices = all_indices # List of dictionaries of vertices, normal, and textures for each face
        self.indices = indices
        self.generate_face_normals()
        self.normals = np.asarray(self.normals)
        

    def generate_face_normals(self):
        for i in range(len(self.all_indices)):
            if self.all_indices[i]["vn"] == []:
                vertex_indices = self.all_indices[i]["v"]
                face = [self.points[j] for j in vertex_indices]
                normal = get_face_normal(face)
                nx, ny, nz, nw = normal
            else:
                nx, ny, nz, nw = np.mean(np.array([self.normals[index] for index in self.all_indices[i]["vn"]]), axis=0)
            
            self.all_indices[i]["vn"] = [nx,ny,nz,nw]

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
        self.cam.update_cam()

        todraw = []
        for face_values in self.all_indices:
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
            pg.draw.polygon(self.screen,(185*color,245*color,185*color), polygon, width = 0)
            # Draws edges on triangles
            pg.draw.polygon(self.screen, "white", polygon, width = 1)
    
    

    # This is a function that technically takes over three arguments, so maybe some of them should have a class
    # this function actually does not need the self argument (it used to and i just didnt remove it)
    # The name of my function is formatted correctly
    
    
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

