from constants import *



def signed_distance(plane_normal:np.ndarray, vertex:np.ndarray, plane_point:np.ndarray) -> np.ndarray:
    """
    What this does...

    Parameters:
    ----------------
    plane_normal : np.ndarray

    vertex : np.ndarray 
    
    plane_point : np.ndarray

    ----------------
    Returns:
    ----------------
    """
    return (plane_normal[0] * (vertex[0] - plane_point[0]) + plane_normal[1] * (vertex[1] - plane_point[1]) + plane_normal[2] * (vertex[2] - plane_point[2]))

def linear_interpolation(p1: np.ndarray, p2: np.ndarray, dist: float) -> np.ndarray:
    """
    What this does...

    Parameters:
    ----------------
    p1 : np.ndarray 

    p2 : np.ndarray 
    
    dist : float

    ----------------
    Returns:
    ----------------
    """
    p1x,p1y,p1z,p1w = p1 
    p2x,p2y,p2z,p2w = p2
    #denom = ((p1x-p2x)**2+(p1y-p1y)**2+(p1z-p2z)**2)**0.5
    newx = p1x + dist * (p2x - p1x)
    newy = p1y + dist * (p2y - p1y)
    newz = p1z + dist * (p2z - p1z)

    return np.array([newx, newy, newz, 1])

def line_plane_intersect(plane_normal: np.ndarray, p1: np.ndarray, p2: np.ndarray, plane_point: np.ndarray) -> np.ndarray:
    """
    What this does...

    Parameters:
    ----------------
    plane_normal : np.ndarray

    p1 : np.ndarray 

    p2 : np.ndarray 
    
    plane_point : np.ndarray

    ----------------
    Returns:
    ----------------
    """
    p1x,p1y,p1z,p1w = p1 
    p2x,p2y,p2z,p2w = p2
    # Normal values
    nx, ny, nz, nw = plane_normal
    slopex, slopey, slopez = p1x-p2x,p1y-p2y,p1z-p2z
    mag = ((slopex**2) + (slopey**2) + (slopez**2)) ** 0.5
    if mag == 0: mag = 1
    ux, uy, uz = slopex/mag, slopey/mag, slopez/mag
    dot = ux * nx + uy * ny + uz * nz
    if dot == 0: dot = 1
    d = ((plane_point[0] - p2x) * nx + (plane_point[1] - p2y) * ny + (plane_point[2] - p2z) * nz) / dot
    x = p2x + ux * d
    y = p2y + uy * d
    z = p2z + uz * d

    return np.array([x, y, z, 1.0])

def get_face_normal_vectorized(mesh: np.ndarray) -> np.ndarray:
    """
    What this does...

    Parameters:
    ----------------
    face : np.ndarray

    ----------------
    Returns:
    ----------------
    """
    # Note: this is if the vertices are in clockwise order. If they are not, then the cross product should be the other way
    if mesh.size == 0:
        return np.zeros((1,4))
    
    if mesh.ndim == 3:
        normals = []
        edge1 = mesh[:,2,:3] - mesh[:,0,:3]
        edge2 = mesh[:,1,:3] - mesh[:,0,:3]
        normals = np.cross(edge1, edge2, axis=1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms != 0, norms, 1)
        normals = np.hstack([normals, np.zeros((normals.shape[0], 1))])
        return normals

    elif mesh.ndim == 2:
        x1, y1, z1, w1 = (mesh[2] - mesh[0])
        x2, y2, z2, w2 = (mesh[1] - mesh[0])
        # returns the vector perpendicular to the face, which I will compare with either the camera direction or the light direction
        normal = [(y1 * z2 - z1 * y2), 
                (z1 * x2 - x1 * z2), 
                (x1 * y2 - y1 * x2), 0]
        nx, ny, nz, nw = normal
        norm = (nx ** 2 + ny ** 2 + nz ** 2)**0.5
        if norm != 0:
            normal = [nx/norm, ny/norm, nz/norm, 0]
        return normal
    
    else:
        return
    """for face in mesh:
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
        normals.append(normal)"""
    
def angle_between_points(p1:np.ndarray, p2:np.ndarray, p3:np.ndarray) -> float:
    """Returns angle between points in radians"""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    print(dot)
    magv1 = np.linalg.norm(v1)
    magv2 = np.linalg.norm(v2)
    return np.arccos(dot/(magv1 * magv2))

def angle_between_vectors(v1:np.ndarray, v2:np.ndarray) -> float:
    dot = np.dot(v1, v2)
    magv1 = np.linalg.norm(v1)
    magv2 = np.linalg.norm(v2)
    return np.arccos(dot/(magv1 * magv2))

def normalize(vertex: np.ndarray, tolerance: float=0.00001):
    mag2 = sum(n * n for n in vertex[:-1])
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = np.array([n / mag for n in vertex[:-1]])
        v = np.append(v,1)
    return v


def zordermesh(mesh: np.ndarray) -> np.ndarray:
    """
    Orders a mesh by the z-coordinates of its points

    Parameters:
    ----------------
    mesh : np.ndarray

    ----------------
    Returns:
    ----------------
    The mesh ordered by its z-coordinates
    """
    if mesh.size == 0:
        return mesh

    if mesh.shape[1] > 3:
        zmeans = np.mean(mesh[:, :-1, 2], axis=1)
        reordered_indices = np.argsort(zmeans)[::-1]

    else:
        zmeans = np.mean(mesh[:, :, 2], axis=1)
        reordered_indices = np.argsort(zmeans)[::-1]

    return mesh[reordered_indices]

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
    """
    What this does...

    Parameters:
    ----------------
    triangle : np.ndarray

    plane : np.ndarray 

    plane_point : np.ndarray

    ----------------
    Returns:
    ----------------
    """
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

def clip_triangle(triangle: np.ndarray, plane: np.ndarray, plane_point: np.ndarray = np.array([0,0,0,0])) -> list:
    """
    What this does...

    Parameters:
    ----------------
    triangle : np.ndarray

    plane : np.ndarray 

    plane_point : np.ndarray

    ----------------
    Returns:
    ----------------
    """
    in_bounds = [None, None, None]
    num_out = 0
    vertices = triangle[:-1]
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



def to_pygame(array: np.ndarray) -> np.ndarray:
    """
    Takes an array of any dimension (ndim > 1) and converts its last dimension values to pygame coordinates

    Parameters:
    ----------------
    array : np.ndarray
        An array, which contains point coordinates in the last dimension
    ----------------
    Returns:
    ----------------
    Pygame Coordinates
    """
    return (array[...,:2] * PYGAME_WINDOW_SCALE) + PYGAME_WINDOW_ORIGIN

def to_frustum(points: np.ndarray, maxval:float, shift:np.ndarray = np.array([0,0,0,0])) -> np.ndarray:
    """
    Takes a vector of dimension 1 or 2 and converts its points to frustum coordinates

    Parameters:
    ----------------
    points : np.ndarray
        The points to convert
    maxval : float
        Maximum value of the array to normalize the points by
    shift : np.ndarray
        Vector to shift the points by
    ----------------
    Returns:
    ----------------
    Viewing Frustum Coordinates
    """
    if points.ndim == 1:
        z = NEAR_Z * ((points[2])/maxval) + shift[2]
        x = NEAR_Z * ((points[0])/maxval)
        y = NEAR_Z * ((points[1])/maxval)
        return np.array([x,y,z,points[3]])
    else:
        points[:,:3] *= NEAR_Z/maxval
        points[:,:3] += shift[:3]
        return points

def ortho_project_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    What this does...

    Parameters:
    ----------------
    polygon : np.ndarray

    ----------------
    Returns:
    ----------------
    """
    n = polygon.shape[0]
    normal = get_face_normal_vectorized(polygon)
    basisU = polygon[1] - polygon[0]
    basisU /= np.linalg.norm(basisU)
    basisV = np.cross(normal, basisU)
    
    projected_points = np.zeros((n, 2))
    for i in range(n):
        p_prime = polygon[i] - polygon[0]
        x = np.dot(p_prime, basisU)
        y = np.dot(p_prime, basisV)
        projected_points[i] = (x,y)
    return projected_points



def triangulate_mesh(mesh: np.ndarray) -> np.ndarray:
    """
    What this does...

    Parameters:
    ----------------
    mesh : np.ndarray

    ----------------
    Returns:
    ----------------
    """
    ortho_mesh = np.array([ortho_project_polygon(polygon) for polygon in mesh])
    return

def ear_triangulate(polygon: np.ndarray) -> np.ndarray:
    """
    This triangulates a 2D polygon using the ear method

    Parameters:
    ----------------
    polygon : np.ndarray
        An array of points in 2D coordinates

    ----------------
    Returns:
    ----------------
    """
    triangles = []
    n = len(polygon)
    prev = polygon[-1]
    cur = polygon[0]

    i = 0
    count = 0
    while n > 0:
        nextitem = polygon[(i+1)%n]
        v1 = prev - cur
        v2 = nextitem - cur
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product < 0:
            return
        angle = angle_between_vectors(v1, v2)
        if angle == math.pi:
            return
        


        prev = cur
        cur = nextitem

    return


p1 = np.array([1,0])
p2 = np.array([0,0])
p3 = np.array([0,-1])

v1 = p1 - p2
v2 = p3 - p2
cross_product = v1[0] * v2[1] - v1[1] * v2[0]
#print(angle_between_vectors(v1, v2), cross_product)


class Triangulate3D:
    def __init__(self) -> None:
        pass

    def sphere(self):
        pass

    def ellipsoid(self):
        pass

    def mesh(self, mesh: np.ndarray) -> np.ndarray:
        pass


