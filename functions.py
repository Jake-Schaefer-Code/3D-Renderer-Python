import numpy as np
import random
from constants import *

def signed_distance(plane_normal: np.ndarray, vertex: np.ndarray, plane_point: list=[0,0,0,0]) -> float:
    return plane_normal[0] * (vertex[0] - plane_point[0]) + plane_normal[1] * (vertex[1] - plane_point[1]) + plane_normal[2] * (vertex[2] - plane_point[2])

def line_plane_intersect(plane_normal: np.ndarray, p1: np.ndarray, p2: np.ndarray, plane_point: np.ndarray=np.array([0,0,0,0])) -> np.ndarray:
    p1x,p1y,p1z,p1w = p1 
    p2x,p2y,p2z,p2w = p2
    # Normal values
    nx, ny, nz, nw = plane_normal
    slopex, slopey, slopez = p1x-p2x,p1y-p2y,p1z-p2z
    mag = ((slopex**2) + (slopey**2) + (slopez**2)) ** 0.5
    if mag == 0: mag = 1e-6
    ux, uy, uz = slopex/mag, slopey/mag, slopez/mag
    dot = ux * nx + uy * ny + uz * nz
    if dot == 0: dot = 1e-6
    d = ((plane_point[0] - p2x) * nx + (plane_point[1] - p2y) * ny + (plane_point[2] - p2z) * nz) / dot
    return np.array([p2x + ux * d, p2y + uy * d, p2z + uz * d,1])

def get_face_normal(face):
    # Note: this is if the vertices are in clockwise order. If they are not, then the cross product should be the other way
    x1, y1, z1, w1 = (face[2] - face[0])
    x2, y2, z2, w2 = (face[1] - face[0])
    # returns the vector perpendicular to the face, which I will compare with either the camera direction or the light direction
    normal = [(y1 * z2 - z1 * y2), 
              (z1 * x2 - x1 * z2), 
              (x1 * y2 - y1 * x2), 0]
    nx, ny, nz, nw = normal
    norm = (nx ** 2 + ny ** 2 + nz ** 2)**0.5
    if norm != 0:
        normal = [nx/norm, ny/norm, nz/norm, 0]
    return normal

def to_pygame(point: list) -> list:
    return [(point[0] * HALFWIDTH) + HALFWIDTH, (point[1] * HALFHEIGHT) + HALFHEIGHT]

def translate(matrix: np.ndarray, trans_vector: list) -> np.ndarray:
    return np.array([[1,0,0,-trans_vector[0]],
                     [0,1,0,-trans_vector[1]],
                     [0,0,1,-trans_vector[2]],
                     [0,0,0,1 ]]) @ matrix

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