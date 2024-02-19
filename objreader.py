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
    vertices = []
    vertex_indices = []
    normal_indices = []
    texture_indices = []
    indices = []
    normals = []
    textures = []
    f = open(fileName)
    maxval = 1
    for line in f:
        vals = line.split()
        if len(vals) != 0:
            key = vals[0]
            # Vertices
            if key == "v":
                vertex = [float(vals[1]),float(vals[2]), float(vals[3]),1]
                abs = np.abs(vertex)
                maxval = max(np.max(abs), maxval)
                #vertex = normalize(vertex)
                vertices.append(vertex)

            # Indices of vertices for the faces
            elif key =="f":
                # This index-n changes between files its weird
                # TODO maybe add a method to find the lowest index in a file and subtract from this
                #face = np.array([int(index) for index in vals[1:]]) - 1
                face_coords = []
                texture_coords = []
                normal_coords = []
                #keys = ["v","vt","vn"]
                coord_dict = {"v":[],"vt":[],"vn":[],"face_normal":[]}
                for val in vals[1:]:
                    elements = val.split('/')
                    if len(elements) == 3:
                        
                        coord_dict["vt"].append(elements[1])
                        coord_dict["vn"].append(int(elements[2])-1)
                    coord_dict["v"].append(int(elements[0])-1)
                    face_coords.append(int(elements[0]))
                
                face = np.array(face_coords)-1
                
                #face = np.array([int(index) for index in face_coords]) - 1

                #mesh.append(np.array([vertices[index-1] for index in face]))

                vertex_indices.append(face[:3])
                texture_indices.append(texture_coords)
                normal_indices.append(normal_coords)
                indices.append(coord_dict)


            # Normals of faces
            elif key == "vn":
                normal = [float(vals[1]), float(vals[2]), float(vals[3]), 0]
                normals.append(normal)
            
            # Texture coordinates
            elif key == "vt":
                if len(vals[1:]) == 2:
                    texture_coords = [float(vals[1]), float(vals[2])]
                else:
                    texture_coords = [float(vals[1]), float(vals[2]), float(vals[3])]
                textures.append(texture_coords)
            

    f.close()

    return np.asarray(vertex_indices), np.asarray(vertices), maxval, normals, np.asarray(indices)

