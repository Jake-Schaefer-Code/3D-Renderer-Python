import numpy as np
import math
import pygame
import random
from collections import deque
import cProfile,pstats
import cython
import numba

from quaternions import *
from functions import *
from constants import *
from camera import *



class Object:
    def __init__(self, points: np.ndarray, position: np.ndarray, indices: np.ndarray, camera: Camera, 
                 screen: pygame.display, normals: np.ndarray, all_indices: np.ndarray, maxval: float=1.0, tofrust: bool = True):
        self.cam = camera
        # Local Origin
        self.screen = screen
        self.maxval = maxval
        


        self.points = np.array([self.to_frustum(p) for p in points]) if tofrust else np.array([p for p in points])
        # TODO move this elsewhere?
        self.points = np.array([p+position for p in self.points])
        
        self.position = position
        self.center = position

        self.numpoints = len(points)
        self.polygons = indices
        self.numfaces = len(self.polygons)
        self.lightdir = np.array([0,0,-1,0])
        
        self.planes = PLANES
        self.plane_points = PLANE_POINTS
        self.normals = normals
        self.all_indices = all_indices
        #self.indices = indices
        self.generate_face_normals()
    

    def generate_face_normals(self):
        for i in range(len(self.all_indices)):
            if self.all_indices[i]["vn"] == []:
                vertex_indices = self.all_indices[i]["v"]
                face = [self.points[j] for j in vertex_indices]
                normal = get_face_normal(face)
                nx = normal[0]
                ny = normal[1]
                nz = normal[2]
                nw = normal[3]
                #nx, ny, nz, nw = get_face_normal(face)
            else:
                nx, ny, nz, nw = np.mean(np.array([self.normals[index] for index in self.all_indices[i]["vn"]]), axis=0)
            self.all_indices[i]["face_normal"] = [nx,ny,nz,nw]
            self.normals.append([nx,ny,nz,nw])

    # converts to frustum coordinates
    def to_frustum(self, point: np.ndarray):
        z = NEAR_Z*((point[2])/self.maxval) + 2* self.cam.center[2]
        x = NEAR_Z*((point[0])/self.maxval)
        y = -NEAR_Z*((point[1])/self.maxval)
        return np.array([x,y,z,point[3]])

    def prepare_mesh(self) -> list:
        transformpoints = [self.cam.transform_point(point) for point in self.points]
        todraw = []
        for face_indices in self.all_indices:
            face = [transformpoints[j] for j in face_indices["v"]]
            normal = self.cam.transform_point(face_indices["face_normal"])
            nx = normal[0]
            ny = normal[1]
            nz = normal[2]
            #nx,ny,nz,nw = self.cam.transform_point(face_indices["face_normal"])

            # back-face culling: only draw if face is facing camera i.e. if normal is facing in negative direction
            # This is the normal dotted with the view vector pointing from the polygon's surface to the camera's position
            coincide = nx * face[1][0] + ny * face[1][1] + nz * face[1][2]
            if coincide < 0:
                # TODO FIX
                norm_color_dot = (nx*self.lightdir[0] + 
                                  ny*self.lightdir[1] + 
                                  nz*self.lightdir[2]) / 2 
                
                # Setting the last index of the face to the colorval
                face.append( -norm_color_dot/10 if norm_color_dot < 0 else norm_color_dot)
                #face[-1] = -norm_color_dot/10 if norm_color_dot < 0 else norm_color_dot
                todraw.append(face)
        
        # Calls to functions that clip the mesh and z-order it
        todraw = zordermesh(todraw)
        todraw = self.clip_mesh(todraw)
        return todraw

    def draw4(self) -> None:
        draw_mesh = self.prepare_mesh()
        for face in draw_mesh:
            # TODO maybe do the colors here instead of above
            # TODO maybe have references to an array of colovals
            color = face[-1]
            # Projects points and converts them to pygame coordinates
            polygon = [to_pygame(self.cam.perspective_projection(p)) for p in face[:-1]]
            #polygon = Projected_List(face[:-1], self.cam)
            
            # Draws Polygons
            pygame.draw.polygon(self.screen,(185*color,245*color,185*color), polygon, width = 0)
            # Draws edges on triangles
            #pygame.draw.polygon(self.screen,"white", polygon, width = 1)

    """def draw3(self) -> None:
        #self.transformpoints = []
        #self.transformpoints = Transformed_List(self.points, self.cam)
        self.transformpoints = [self.cam.transform_point(point) for point in self.points]
        todraw = []

        for face_indices in self.all_indices:
            face = [self.transformpoints[j] for j in face_indices["v"]]+[0.0]
            normal = self.cam.transform_point(face_indices["face_normal"])
            nx = normal[0]
            ny = normal[1]
            nz = normal[2]
            #nx,ny,nz,nw = self.cam.transform_point(face_indices["face_normal"])
            # only draw if face is facing camera i.e. if normal is facing in negative direction

            coincide = nx * face[1][0] + ny * face[1][1] + nz * face[1][2]
            if coincide < 0:
                norm_color_dot = (nx*self.lightdir[0] + 
                                  ny*self.lightdir[1] + 
                                  nz*self.lightdir[2]) / 2
                
                # Setting the last index of the face to the colorval
                face[-1] = -norm_color_dot/10 if norm_color_dot < 0 else norm_color_dot
                todraw.append(face)
        
        # Calls to functions that clip the mesh and z-order it
        todraw = zordermesh(todraw)
        draw_mesh = self.clip_mesh(todraw)
        #todraw = self.clip_mesh(todraw)
        #draw_mesh = self.zordermesh(todraw)

        for face in draw_mesh:
            # TODO maybe do the colors here instead of above
            # TODO maybe have references to an array of colovals
            color = face[-1]
            # Projects points and converts them to pygame coordinates
            polygon = [to_pygame(self.cam.perspective_projection(p)) for p in face[:-1]]
            #polygon = Projected_List(face[:-1], self.cam)
            
            # Draws Polygons
            pygame.draw.polygon(self.screen,(185*color,245*color,185*color), polygon, width = 0)
            # Draws edges on triangles
            #pygame.draw.polygon(self.screen,"white", polygon, width = 1)"""
    
    def clip_mesh(self, mesh: list[list]):
        triangle_queue = deque(mesh) 
        """for i in range(len(self.cam.planes)):
            plane = self.cam.planes[i]"""
        for plane in self.cam.planes:
            for _ in range(len(triangle_queue)):
                polygon = triangle_queue.popleft()
                if len(polygon) == 4: # Is triangle: 3 vertices and color
                    new_triangles = self.clip_triangle(polygon, plane)
                    """if(plane == np.array([0, 0, 1, 0])).all():
                        new_triangles = self.clip_triangle(polygon, plane, np.array([0,0,NEAR_Z,0]))
                    else:
                        new_triangles = self.clip_triangle(polygon, plane)"""
                    triangle_queue.extend(new_triangles)
                else:
                    triangle_queue.append(polygon)
        return list(triangle_queue)

    # This is a function that technically takes over three arguments, so maybe some of them should have a class
    # this function actually does not need the self argument (it used to and i just didnt remove it)
    # The name of my function is formatted correctly
    
    def clip_triangle(self, triangle: np.ndarray, plane: np.ndarray, plane_point: list = [0,0,0,0]) -> list:
        in_bounds = [None,None,None]
        num_out = 0

        # Ignoring homogenous coordinate
        for i in range(3):
            vertex = triangle[i]
            if signed_distance(plane, vertex, plane_point) < 0: num_out += 1
            else: in_bounds[i] = vertex
        if num_out == 0: return [triangle]
        
        # if one point is OOB, then make 2 new triangles
        elif num_out == 1:
            new_points = []
            color = triangle[-1]
            for i in range(3):
                if in_bounds[i] is not None:
                    new_points.append(in_bounds[i])
                else:
                    new_points.append(line_plane_intersect(plane, triangle[(i-1)%3], triangle[i], plane_point))
                    new_points.append(line_plane_intersect(plane, triangle[(i+1)%3], triangle[i], plane_point))
                    
            triangle1 = [new_points[0],new_points[1],new_points[2],color]
            triangle2 = [new_points[0],new_points[2],new_points[3],color]
            return [triangle1, triangle2]
        
        elif num_out == 2:
            #new_vertices = vertices.copy()
            for i in range(3):
                if in_bounds[i] is not None:
                    vertices = triangle
                    # intersection of plane and line from current vertex to both OOB vertices
                    vertices[(i+1)%3] = line_plane_intersect(plane, triangle[(i+1)%3], triangle[i], plane_point)
                    vertices[(i-1)%3] = line_plane_intersect(plane, triangle[(i-1)%3], triangle[i], plane_point)
                    #new_vertices[(i+1)%3] = line_plane_intersect(plane, vertices[(i+1)%3], triangle[i],plane_point)
                    #new_vertices[(i-1)%3] = line_plane_intersect(plane, vertices[(i-1)%3], triangle[i],plane_point)
                    #new_triangle = np.array([new_vertices[0], new_vertices[1], new_vertices[2], triangle[-1]], dtype=object)
            #return [new_triangle]
            return [vertices]
        
        else: return []
        # There are no hidden side effects in this function - it returns what it is supposed to and modifies nothing else
        # There are lots of return statements

class Transformed_List(list):
    """def __new__(cls, iterable, camera):
        #transformed_iterable = [camera.transform_point(point) for point in iterable]
        return super().__new__(cls)"""
    def __init__(self, iterable, camera):
        transformed_iterable = [camera.transform_point(point) for point in iterable]
        super().__init__(transformed_iterable)




def zordermesh(mesh: list) -> list:
    mesh_size = len(mesh)
    means = [0 for _ in range(mesh_size)]
    for i in range(mesh_size):
        length = len(mesh[i][:-1])
        mean = 0
        for j in range(length):
            mean+=mesh[i][j][2]
        mean /= length
        means[i] = mean

    if means == []:
        return mesh
    #for i in range(len(mesh3)):
    #   means.append(np.mean(mesh3[i]))

    #means = np.mean(mesh3, axis=1)
    #sorted_indices = sorted(enumerate(means), key=lambda x: x[1])
    #reorder = [index for index, _ in sorted_indices]
    reorder = np.argsort(means)[::-1]
    return [mesh[reorder[i]] for i in range(mesh_size)]