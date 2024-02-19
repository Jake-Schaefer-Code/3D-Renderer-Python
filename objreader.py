from typing import Any
import numpy as np
import math


class OBJ:
    def __init__(self, fileName) -> None:
        self.vertices: list[list[float]] = [] 
        self.faces: list[list[int]] =  []
        self.normals: list[list[float]] = []
        self.textures: list[list[float]]= []
        self.fileName = fileName
        self.maxval = 1
        self.lines = self.create_line_generator()

    def read_obj(self):
        with open(self.fileName) as f:
            for line in f:
                vals = line.split()

    def create_line_generator(self):
        file = open(self.fileName, mode='r', encoding=self.encoding)

        for line in file:
            yield line

        file.close()

def normalize(vertex, tolerance=0.00001):
    mag2 = sum(n * n for n in vertex[:-1])
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = np.array([n / mag for n in vertex[:-1]])
        v = np.append(v,1)
    return v


def read_obj(fileName: str):
    #indices = []
    vertices = []
    vertex_indices = []
    normals = []
    textures = []
    texture_indices = []
    normal_indices = []
    with open(fileName) as f:
        for line in f:
            vals = line.split()
            if len(vals) == 0:
                continue
            elif vals[0] == '#':
                continue
            
            key = vals[0]

            # Vertices
            if key == "v":
                vertices.append(np.array([float(vals[1]), float(vals[2]), float(vals[3]), 1]))

            # Indices of vertices for the faces
            elif key =="f":
                # This index-n changes between files its weird
                # TODO maybe add a method to find the lowest index in a file and subtract from this
                face_coords = [[],[],[]]
                for val in vals[1:]:
                    elements = val.split('/')
                    if len(elements) >= 2:
                        face_coords[1].append(elements[1])
                    else:
                        face_coords[1].append(-1)

                    if len(elements) >= 3:
                        face_coords[2].append(elements[2])
                    else:
                        face_coords[2].append(-1)
                    
                    face_coords[0].append(int(elements[0]))               
                
                vertex_indices.append(face_coords[0])
                texture_indices.append(face_coords[1])
                normal_indices.append(face_coords[2])

            # Normals of faces
            elif key == "vn":
                normals.append([float(vals[1]), float(vals[2]), float(vals[3]), 0])
            
            # Texture coordinates
            elif key == "vt":
                textures.append(np.float64(vals[1:]))

    minIndex = np.min(vertex_indices)
    colors = np.zeros((len(vertex_indices), 3))
    dtype = [('v', 'i4', (3,)), ('vn', 'i4', (3,)), ('vt', 'i4', (3,)), ('c', 'f4', (3,))]
    component_array = np.empty(len(vertex_indices), dtype=dtype)
    component_array['v'] = np.array(vertex_indices, dtype='i4') - minIndex
    component_array['vn'] = np.array(normal_indices, dtype='i4') - minIndex
    component_array['vt'] = np.array(texture_indices, dtype='i4') - minIndex 
    component_array['c'] =  colors
    return np.array(vertices), normals, component_array

