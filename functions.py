import numpy as np
import random
from constants import *

def signed_distance(plane_normal: np.ndarray, vertex: np.ndarray, plane_point: list=[0,0,0,0]) -> float:
    return plane_normal[0] * (vertex[0] - plane_point[0]) + plane_normal[1] * (vertex[1] - plane_point[1]) + plane_normal[2] * (vertex[2] - plane_point[2])

def linear_interpolation(p1, p2, dist):
    p1x,p1y,p1z,p1w = p1 
    p2x,p2y,p2z,p2w = p2
    #denom = ((p1x-p2x)**2+(p1y-p1y)**2+(p1z-p2z)**2)**0.5
    newx = p1x + dist * (p2x - p1x)
    newy = p1y + dist * (p2y - p1y)
    newz = p1z + dist * (p2z - p1z)

    return np.array([newx, newy, newz, 1])


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


### VECTORIZED ###
def vectorized_zordermesh(mesh: np.ndarray) -> np.ndarray:
    if mesh.size == 0:
        return mesh
    zmeans = np.mean(mesh[:, :-1, 2], axis=1)
    reordered_indices = np.argsort(zmeans)[::-1]
    return mesh[reordered_indices]

def vectorized_zordermesh2(mesh: np.ndarray) -> np.ndarray:
    if mesh.size == 0:
        return mesh
    zmeans = np.mean(mesh[:, :, 2], axis=1)
    reordered_indices = np.argsort(zmeans)[::-1]
    return reordered_indices


def new_clip(triangle: np.ndarray, plane: np.ndarray, plane_point: np.ndarray = np.array([0,0,0,0])) -> list:
    #previous_distance = signed_distance(plane, triangle[0], plane_point)
    distances = [signed_distance(plane, triangle[i], plane_point) for i in range(3)]
    condition = [distances[0] < 0 ,distances[1] < 0 ,distances[2] < 0]

    """if distances[0] < 0 and distances[1] < 0 and distances[2] < 0:
        return []
    if not distances[0] < 0 and not distances[1] < 0 and not distances[2] < 0:
        return [triangle]"""
    
    if condition[0] and condition[1] and condition[2]:
        return []
    if not condition[0] and not condition[1] and not condition[2]:
        return [triangle]
    
    #v0, v1, v2, c = triangle
    # Second vertex OOB and first is not
    if condition[1] and not condition[0]:
        next_distance = distances[2]
        #v3 = v0 # IB
        #v0 = v1 # OOB
        #v1 = v2 # 
        #v2 = v3 # IB
        v0index = 1
        v1index = 2
        v2index = 0

    elif condition[2] and not condition[1]:
        next_distance = distances[0]
        #v3 = v2 # OOB
        #v2 = v1 # IB
        #v1 = v0
        #v0 = v3 # OOB
        v0index = 2
        v1index = 0
        v2index = 1
        
    else:
        next_distance = distances[1]
        # v0 OOB and v2 IB
        v0index = 0
        v1index = 1
        v2index = 2
    
    v0, v1, v2, c = triangle[v0index], triangle[v1index], triangle[v2index], triangle[-1]
    v3 = linear_interpolation(v0, v2, np.abs(distances[v0index] / (distances[v0index] - distances[v2index])))
    if next_distance < 0:
        v2 = linear_interpolation(v1, v2, np.abs(distances[v1index] / (distances[v1index] - distances[v2index])))
        return [np.array([v0,v1,v2,c]), np.array([v2,v3,v0,c])]
        #return [np.array([v1,v2,v3,triangle[-1]])]
    else:
        v0 = linear_interpolation(v0, v1, np.abs(distances[v0index] / (distances[v0index] - distances[v1index])))
        return [np.array([v1,v2,v3,c])]
        #return [np.array([v0,v1,v2,triangle[-1]]), np.array([v2,v3,v0,triangle[-1]])]

def vectorized_clip_triangle2(triangle: np.ndarray, plane: np.ndarray, plane_point: np.ndarray = np.array([0,0,0,0])) -> list:
    in_bounds = [None,None,None]
    num_out = 0
    vertices = triangle
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
        for i in range(3):
            if in_bounds[i] is not None:
                new_points.append(in_bounds[i])
            else:
                new_points.append(line_plane_intersect(plane, vertices[(i-1)%3], vertices[i], plane_point))
                new_points.append(line_plane_intersect(plane, vertices[(i+1)%3], vertices[i], plane_point))
                
        triangle1 = np.array([new_points[0],new_points[1],new_points[2]])
        triangle2 = np.array([new_points[0],new_points[2],new_points[3]])
        return [triangle1, triangle2]
    
    # If two points are OOB, then chop off the part of the triangle out of bounds
    elif num_out == 2:
        for i in range(3):
            if in_bounds[i] is not None:
                new_vertices = vertices
                # intersection of plane and line from current vertex to both OOB vertices
                new_vertices[(i+1)%3] = line_plane_intersect(plane, vertices[(i+1)%3], vertices[i], plane_point)
                new_vertices[(i-1)%3] = line_plane_intersect(plane, vertices[(i-1)%3], vertices[i], plane_point)

        triangle = new_vertices
        return [triangle]
    
    else: return []


def vectorized_clip_triangle(triangle: np.ndarray, plane: np.ndarray, plane_point: np.ndarray = np.array([0,0,0,0])) -> list:
    in_bounds = [None,None,None]
    num_out = 0
    vertices = triangle[:-1]
    # Checking each vertex in the triangle
    for i in range(3):
        vertex = vertices[i]
        #print(vertex, signed_distance(plane, vertex, plane_point))
        if signed_distance(plane, vertex, plane_point) < 0: 
            #print(vertex, signed_distance(plane, vertex, plane_point))
            num_out += 1
        
        else: in_bounds[i] = vertex
    if num_out == 0: return [triangle]


    # If one point is OOB, then make 2 new triangles
    elif num_out == 1:
        new_points = []
        color = triangle[-1]
        for i in range(3):
            if in_bounds[i] is not None:
                new_points.append(in_bounds[i])
            else:
                new_points.append(line_plane_intersect(plane, vertices[(i-1)%3], vertices[i], plane_point))
                new_points.append(line_plane_intersect(plane, vertices[(i+1)%3], vertices[i], plane_point))
                
        triangle1 = np.array([new_points[0],new_points[1],new_points[2], color])
        triangle2 = np.array([new_points[0],new_points[2],new_points[3], color])
        return [triangle1, triangle2]
    
    # If two points are OOB, then chop off the part of the triangle out of bounds
    elif num_out == 2:
        for i in range(3):
            if in_bounds[i] is not None:
                new_vertices = vertices
                # intersection of plane and line from current vertex to both OOB vertices
                new_vertices[(i+1)%3] = line_plane_intersect(plane, vertices[(i+1)%3], vertices[i], plane_point)
                new_vertices[(i-1)%3] = line_plane_intersect(plane, vertices[(i-1)%3], vertices[i], plane_point)

        triangle[:-1] = new_vertices
        return [triangle]
    
    else: return []

def vectorized_to_pygame(mesh):
    transform = np.array([HALFWIDTH, HALFHEIGHT])
    return (mesh[:,:,:2] * transform) + transform

def vectorized_to_pygame2(points):
    transform = np.array([HALFWIDTH, HALFHEIGHT])
    return (points[:,:2] * transform) + transform


#print(NEAR_PLANE_WIDTH/2)
#print(signed_distance(np.array([-math.cos(FOVX/2), 0, math.sin(FOVX/2), 0]), np.array([NEAR_PLANE_WIDTH/2, 0, NEAR_Z, 0])))