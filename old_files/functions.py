import numpy as np
import math
from constants import *
import time

def signed_distance(plane_normal, plane_point, vertex):
    dot = plane_normal[0] * (vertex[0] - plane_point[0]) + plane_normal[1] * (vertex[1] - plane_point[1]) + plane_normal[2] * (vertex[2] - plane_point[2])
    return dot

def line_plane_intersect(plane_normal, plane_point, p1, p2):
    slope = p1 - p2
    mag = math.sqrt((slope[0]**2) + (slope[1]**2) + (slope[2]**2) + (slope[3]**2))
    if mag == 0:
        unit_slope = slope/0.000001
    else:
        unit_slope = slope/mag
    
    # parametric value
    #dot = unit_slope[0] * plane_normal[0] + unit_slope[1] * plane_normal[1] +  unit_slope[2] * plane_normal[2]
    #dot_top = point[0] * plane_normal[0] + point[1] * plane_normal[1] + point[2] * plane_normal[2]
    dot = np.dot(unit_slope, plane_normal)
    if dot == 0:
        d = np.dot((plane_point - p2), plane_normal) / 0.000001
    else:
        d = np.dot((plane_point - p2), plane_normal) / dot
    return p2 + unit_slope * d

def get_face_normal(face):
    # Note: this is if the vertices are in clockwise order. If they are not, then the cross product should be the other way
    x1, y1, z1, w1 = (face[2] - face[0])
    x2, y2, z2, w2 = (face[1] - face[0])
    # returns the vector perpendicular to the face, which I will compare with either the camera direction or the light direction
    normal = [y1 * z2 - z1 * y2, 
                z1 * x2 - x1 * z2, 
                x1 * y2 - y1 * x2, 
                0]
    nx, ny, nz, nw = normal
    norm = math.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    if norm != 0:
        normal = [nx/norm, ny/norm, nz/norm, 0]
    return normal


def to_pygame(point: np.ndarray) -> np.ndarray:
    return ((point * PYGAME_WINDOW_SCALE) + PYGAME_WINDOW_ORIGIN)[:-2]

def zordermesh(mesh: list) -> list:
    mesh_size = len(mesh)
    means = [0 for _ in range(mesh_size)]
    for i in range(mesh_size):
        length = len(mesh[i][0])
        mean = 0
        for j in range(length):
            mean+=mesh[i][0][j][2]
        mean /= length
        means[i] = mean

    if means == []:
        return mesh
    
    reorder = np.argsort(means)[::-1]
    return [mesh[reorder[i]] for i in range(mesh_size)]


def clip_triangle(triangle: np.ndarray, plane: np.ndarray, plane_point: list = [0,0,0,0]) -> list:
    in_bounds = [None,None,None]
    num_out = 0
    vertices = triangle[0]
    
    # Checking each vertex in the triangle
    for i in range(3):
        vertex = vertices[i]
        if signed_distance(plane, vertex, plane_point) < 0: 
            num_out += 1
        
        else: in_bounds[i] = vertex
    if num_out == 0: return [triangle]
    
    # If one point is OOB, then make 2 new triangles
    elif num_out == 1:
        new_points = []
        color = triangle[2]
        indices = triangle[1]
        for i in range(3):
            if in_bounds[i] is not None:
                new_points.append(in_bounds[i])
            else:
                new_points.append(line_plane_intersect(plane, vertices[(i-1)%3], vertices[i], plane_point))
                new_points.append(line_plane_intersect(plane, vertices[(i+1)%3], vertices[i], plane_point))
                
        triangle1 = [[new_points[0],new_points[1],new_points[2]], indices, color]
        triangle2 = [[new_points[0],new_points[2],new_points[3]], indices, color]
        return [triangle1, triangle2]
    
    # If two points are OOB, then chop off the part of the triangle out of bounds
    elif num_out == 2:
        for i in range(3):
            if in_bounds[i] is not None:
                new_vertices = vertices
                # intersection of plane and line from current vertex to both OOB vertices
                new_vertices[(i+1)%3] = line_plane_intersect(plane, vertices[(i+1)%3], vertices[i], plane_point)
                new_vertices[(i-1)%3] = line_plane_intersect(plane, vertices[(i-1)%3], vertices[i], plane_point)

        triangle[0] = new_vertices
        return [triangle]
    
    else: return []



def to_frustum_point(point: np.ndarray, maxval: float, shift: np.ndarray = np.array([0,0,0,0])) -> np.ndarray:
    z = NEAR_Z * ((point[2])/maxval) + shift[2]
    x = NEAR_Z * ((point[0])/maxval)
    y = NEAR_Z * ((point[1])/maxval)
    return np.array([x,y,z,point[3]])
