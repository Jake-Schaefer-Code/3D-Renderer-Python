from typing import Any
import numpy as np
import math


class OBJFile:
    def __init__(self, filePath: str) -> None:
        """
        What this does...

        Parameters:
        ----------------
        fileName : str

        ----------------
        Returns:
        ----------------
        np.ndarray, np.ndarray, np.ndarray
        """
        self.vertices = [] 
        self.vertex_indices = []
        self.faces =  []

        self.normals = []
        self.normal_indices = []

        self.textures = []
        self.texture_indices = []

        self.component_array = None

        self.filePath = filePath

        self.maxval = 1

        
        self.lines = self.create_line_generator()

    def read_obj(self):
        """
        What this does...

        Returns:
        ----------------
        np.ndarray, np.ndarray, np.ndarray
        """
        with open(self.filePath) as f:
            for line in f:
                vals = line.split()
                if len(vals) == 0:
                    continue
                elif vals[0] == '#':
                    continue
                
                key = vals[0]

                # Vertices
                if key == "v":
                    self.vertices.append(np.array([float(vals[1]), float(vals[2]), float(vals[3]), 1]))

                # Indices of vertices for the faces
                elif key =="f":
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
                    
                    self.vertex_indices.append(face_coords[0])
                    self.texture_indices.append(face_coords[1])
                    self.normal_indices.append(face_coords[2])

                # Normals of faces
                elif key == "vn":
                    self.normals.append([float(vals[1]), float(vals[2]), float(vals[3]), 0])
                
                # Texture coordinates
                elif key == "vt":
                    self.textures.append(np.float64(vals[1:]))

        minIndex = np.min(self.vertex_indices)
        colors = np.zeros((len(self.vertex_indices), 3))
        dtype = [('v', 'i4', (3,)), ('vn', 'i4', (3,)), ('vt', 'i4', (3,)), ('c', 'f4', (3,))]
        component_array = np.empty(len(self.vertex_indices), dtype=dtype)
        component_array['v'] = np.array(self.vertex_indices, dtype='i4') - minIndex
        component_array['vn'] = np.array(self.normal_indices, dtype='i4') - minIndex
        component_array['vt'] = np.array(self.texture_indices, dtype='i4') - minIndex 
        component_array['c'] =  colors
        self.component_array = component_array
        self.vertices = np.array(self.vertices)
        self.normals = np.array(self.normals)
        self.maxval = np.max(np.abs(self.vertex_indices))
        return self.vertices, self.normals, self.component_array

    def create_line_generator(self):
        file = open(self.fileName, mode='r', encoding=self.encoding)

        for line in file:
            yield line

        file.close()

    


