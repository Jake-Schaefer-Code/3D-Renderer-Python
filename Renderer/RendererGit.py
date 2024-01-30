import numpy as np
import math
import pygame
import random
import time
from collections import deque
import cProfile,pstats
import cython
import numba

from quaternions import *
from objreader import read_obj
from functions import *
from constants import *
#from rayMarching3D import *
#from Objects import *   

# more .obj files https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html

# TODO
    # Create index and staging buffer data structures maybe?
    # UV coordinates for textures - divide them by position rather than have them be their own entity
    # Terrain:
        # Render terrain in 8x8 squares where each square is 2 triangles
        # Triangle strips only 3 vertices for first triangle, but every vertex after that builds a new triangle off the last
# TODO add Orthographic projection matrix
# TODO fix so that is able to take other polygons instead of just triangles
# TODO simplify vertex buffer

# Camera is at (0,0) and is always looking down the z axis
class Camera:
    def __init__(self):
        self.pos = np.array([0,0,0,1])
        #self.lightdir = np.array([0,0,-1,0],dtype='float64')
        self.planes = PLANES
        self.fovx = FOVX
        self.fovy = FOVY
        self.third_person = False

        # Distance to near (projection) plane
        self.nearz = NEAR_Z
        self.lbn = NEAR_Z * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), 1.0])
        
        # Distance to far plane
        self.farz = FAR_Z
        self.rtf = NEAR_Z * np.array([math.tan(FOVX/2), math.tan(FOVY/2), FAR_Z/NEAR_Z])

        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = self.z_scale(np.array([0,0,0,1]))
        self.center3 = self.z_scale(np.array([0,0,0,1]))
        
        # The matrix that will be modified through transitions and rotations
        self.matrix = np.identity(4)

        # Initializing perspective projection matrix
        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l), 0, 0, -n*(r+l)/(r-l)],
                                [0, 2*n/(t-b), 0, -n*(t+b)/(t-b)],
                                [0, 0, (n+f)/(f-n), 2*n*f/(n-f)],
                                [0, 0, 1, 0]])
        
        # For future ray tracing
        self.shapes = []
        self.dist_to_nearest = 1
        self.draw_mesh = []

    def update(self):
        self.nearz = WIDTH/(2*math.tan(self.fovx/2))
        self.planes = [np.array([-self.nearz, 0, WIDTH/2, 0]) , # right
          np.array([self.nearz, 0, WIDTH/2, 0]) ,  # left
          np.array([0, -self.nearz, HEIGHT/2, 0]) , # bottom (because y coords are inverted on screen)
          np.array([0, self.nearz, HEIGHT/2, 0])] # top
        self.planes = np.array([plane/math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2) for plane in self.planes])
        self.fovy = 2 * math.atan(ASPECT_RATIO * math.tan(self.fovx/2))
        self.lbn = self.nearz * np.array([-math.tan(self.fovx/2), -math.tan(self.fovy/2), 1.0])
        self.rtf = self.nearz * np.array([math.tan(self.fovx/2), math.tan(self.fovy/2), FAR_Z/self.nearz])
        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l), 0, 0, -n*(r+l)/(r-l)],
                                [0, 2*n/(t-b), 0, -n*(t+b)/(t-b)],
                                [0, 0, (n+f)/(f-n), 2*n*f/(n-f)],
                                [0, 0, 1, 0]])
        
    # Applying transformation matrix
    def transform_point(self, point):
        return self.matrix @ point
    
    # Projecting point onto projection plane
    def perspective_projection(self, point):
        persp_point = self.perspM @ point
        try:
            return persp_point/persp_point[3]
        except:
            return persp_point/0.000001
        """if persp_point[3] == 0:
            return persp_point/0.000001
        else: 
            return persp_point/persp_point[3]"""

    # Camera movement
    def move_cam(self, trans_vector) -> None:
        """if self.key_move:
            self.center3 = translate(np.identity(4), trans_vector) @ self.center3"""
        self.matrix = translate(self.matrix, trans_vector)
    
    # Camera Rotation
    def rotate_cam(self, axis, angle):
        # For third person rotation
        if self.third_person:
            self.move_cam(self.center3)
            self.matrix = q_mat_rot(self.matrix, axis, -angle)
            self.move_cam(-self.center3)
        # First person
        else:
            self.matrix = q_mat_rot(self.matrix, axis, angle)
        
    # converts to canonical viewing space
    def z_scale(self,point):
        n = self.nearz
        f = self.farz
        c1 = 2*f*n/(n-f)
        c2 = (f+n)/(f-n)
        z = c1/(point[2]-c2)
        x = point[0]
        y = -point[1]
        return np.array([x,y,z,point[3]])
                
# TODO make sure separate objects are drawn in order of their distance to camera
class Object:
    def __init__(self, points, position, indices, camera, screen, normals, all_indices, maxval=1, tofrust = True):
        self.cam = camera
        # Local Origin
        self.screen=screen
        self.maxval=maxval
        
        if tofrust:
            self.points = np.array([self.to_frustum(p) for p in points])
        else:
            self.points = np.array([p for p in points])

        self.points = np.array([p+position for p in self.points])
        
        self.position = position
        self.center = position

        self.transformpoints = points

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
    def to_frustum(self,point):
        z = NEAR_Z*((point[2])/self.maxval) + 2* self.cam.center[2]
        x = NEAR_Z*((point[0])/self.maxval)
        y = -NEAR_Z*((point[1])/self.maxval)
        return np.array([x,y,z,point[3]])

    #@cython.locals(nx=cython.float,ny=cython.float,nz=cython.float,nw=cython.float,coincide=cython.float,color=cython.float)

    def draw3(self) -> None:
        self.transformpoints = []
        #self.transformpoints = Transformed_List(self.points, self.cam)
        for point in self.points:
            self.transformpoints.append(self.cam.transform_point(point))
        
        todraw = []
        """self.transform_normals = Transformed_List(self.normals, self.cam)
        for i in range(len(self.all_indices)):
            face = [self.transformpoints[j] for j in self.indices[i]]+[0.0]
            nx,ny,nz,nw = self.transform_normals[i]
            coincide = nx * face[1][0] + ny * face[1][1] + nz * face[1][2]
            if coincide < 0:
                norm_color_dot = (nx*self.lightdir[0] + 
                        ny*self.lightdir[1] + 
                        nz*self.lightdir[2])/2
                
                # Setting the last index of the face to the colorval
                face[-1] = -norm_color_dot/10 if norm_color_dot < 0 else norm_color_dot
                todraw.append(face)"""

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
                        nz*self.lightdir[2])/2
                
                # Setting the last index of the face to the colorval
                face[-1] = -norm_color_dot/10 if norm_color_dot < 0 else norm_color_dot
                todraw.append(face)
        
        # Calls to functions that clip the mesh and z-order it
        todraw = self.clip_mesh(todraw)
        draw_mesh = self.zordermesh(todraw)

        """project_mesh = Projected_Mesh(draw_mesh, self.cam)
        for i in range(len(project_mesh)):
            color = draw_mesh[i][-1]
            polygon = project_mesh[i]
            pygame.draw.polygon(self.screen,(185*color,245*color,185*color), polygon, width = 0)
            pygame.draw.polygon(self.screen,"white", polygon, width = 1)"""

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
            pygame.draw.polygon(self.screen,"white", polygon, width = 1)

    """def draw2(self) -> None:
        self.transformpoints = []
        for point in self.points:
            self.transformpoints.append(self.cam.transform_point(point))

        todraw = []
        for face_indices in self.polygons:
            #TODO Try this later
            #indices = self.all_indices[i]["v"]

            t = [self.transformpoints[j] for j in face_indices]+[0.0]
            #t = [self.transformpoints[i] for i in self.all_indices[i]["v"]]+[0.0]
            #normal_indices = self.all_indices[i]["vn"]
            #normals = np.mean(np.array([self.normals[i] for i in normal_indices]), axis=0)
            #nx, ny, nz = normals[0], normals[1], normals[2]
            nx, ny, nz, nw = get_face_normal(t)
            
            # only draw if face is facing camera i.e. if normal is facing in negative direction
            coincide = nx * t[1][0] + ny * t[1][1] + nz * t[1][2]

            if coincide < 0:
                normdot = (nx*self.lightdir[0] + 
                       ny*self.lightdir[1] + 
                       nz*self.lightdir[2])/2
                
                # TODO fix this
                if normdot < 0:
                    colorval = -normdot/10
                else:
                    colorval = normdot

                t[-1] = colorval
                todraw.append(t)

        # Calls to functions that clip the mesh and z-order it
        todraw = self.clip_mesh(todraw)
        draw_mesh = self.zordermesh(todraw)
        for i in range(len(draw_mesh)):
            # TODO maybe do the colors here instead of above
            t = draw_mesh[i][:-1]
            c = draw_mesh[i][-1]
            
            # Projects points and converts them to pygame coordinates
            t = [to_pygame(self.cam.perspective_projection(p)) for p in t]

            # Draws Triangles
            pygame.draw.polygon(self.screen,(185*c,245*c,185*c), t, width = 0)
            # Draws edges on triangles
            #pygame.draw.polygon(self.screen,"white", t, width = 1)"""

    """def draw(self) -> None:
        for i in range(self.numpoints):
            self.transformpoints[i] = self.cam.transform_point(self.points[i])

        #self.lightdir = self.cam.transform_point(np.array([0,0,-1,0]))
        self.lightdir = np.array([0,0,-1,0])

        #self.center = self.cam.transform_point(self.position)
        self.draw_mesh = []
        for i in range(self.numfaces):
            #indices = self.all_indices[i]["v"]
            t = [self.transformpoints[i] for i in self.polygons[i]]+[0.0]
            #t = [self.transformpoints[i] for i in self.all_indices[i]["v"]]+[0.0]
            nx, ny, nz, nw = get_face_normal(t)
            #normal_indices = self.all_indices[i]["vn"]
            #normals = np.mean(np.array([self.normals[i] for i in normal_indices]), axis=0)
            #nx, ny, nz = normals[0], normals[1], normals[2]
            
            # only draw if face is facing camera i.e. if normal is facing in negative direction
            coincide = nx * t[1][0] + ny * t[1][1] + nz * t[1][2]

            if coincide < 0:
                normdot = (nx*self.lightdir[0] + 
                       ny*self.lightdir[1] + 
                       nz*self.lightdir[2])/2
                
                # TODO fix this
                if normdot < 0:
                    colorval = -normdot/10
                else:
                    colorval = normdot

                t[-1] = colorval
                self.draw_mesh.append(t)

        # Calls to functions that clip the mesh and z-order it
        self.draw_mesh = self.clip_mesh(self.draw_mesh)
        self.draw_mesh = self.zordermesh(self.draw_mesh)
        for i in range(len(self.draw_mesh)):
            # TODO maybe do the colors here instead of above
            t = self.draw_mesh[i][:-1]
            c = self.draw_mesh[i][-1]
            
            # Projects points and converts them to pygame coordinates
            t = [to_pygame(self.cam.perspective_projection(p)) for p in t]

            # Draws Triangles
            pygame.draw.polygon(self.screen,(185*c,245*c,185*c), t, width = 0)
            # Draws edges on triangles
            #pygame.draw.polygon(self.screen,"white", t, width = 1)"""


    def zordermesh(self, mesh):
        #means = [np.mean([m[2] for m in mesh[i][:-1]]) for i in range(len(mesh))]
        means = []
        for i in range(len(mesh)):
            length = len(mesh[i][:-1])
            mean = 0
            for j in range(length):
                mean+=mesh[i][j][2]
            mean /= length
            means.append(mean)

        if means == []:
            return mesh
        #for i in range(len(mesh3)):
        #   means.append(np.mean(mesh3[i]))

        #means = np.mean(mesh3, axis=1)
        #sorted_indices = sorted(enumerate(means), key=lambda x: x[1])
        #reorder = [index for index, _ in sorted_indices]
        reorder = np.argsort(means)[::-1]

        new_mesh = []
        for i in range(len(mesh)):
            new_mesh.append(mesh[reorder[i]])
        return new_mesh
    

    def clip_mesh(self, mesh: list[list]):
        triangle_queue = deque(mesh) 
        #plane_point = np.array([0,0,0,0])  
        for i in range(len(self.cam.planes)):
            plane = self.cam.planes[i]
            for _ in range(len(triangle_queue)):
                polygon = triangle_queue.popleft()
                if len(polygon) == 4: # Is triangle: 3 vertices and color
                    #if  (plane == np.array([0, 0, 1, 0])).all():
                    if any_func(plane, np.array([0, 0, 1, 0])):
                        new_triangles = self.clip_triangle(polygon, plane, np.array([0,0,NEAR_Z,0]))
                    else:
                        new_triangles = self.clip_triangle(polygon, plane)
                    triangle_queue.extend(new_triangles)
                else:
                    triangle_queue.append(polygon)
        return list(triangle_queue)

    # This is a function that technically takes over three arguments, so maybe some of them should have a class
    # this function actually does not need the self argument (it used to and i just didnt remove it)
    # The name of my function is formatted correctly
    #@cython.locals(i=cython.int, in_bounds=list,num_out=cython.int, vertices=list, new_points=list, triangle=list, plane_point=list)
    
    def clip_triangle(self, triangle, plane, plane_point = np.array([0,0,0,0])):
        in_bounds = [None,None,None]
        num_out = 0

        # Ignoring homogenous coordinate
        vertices = triangle#[:-1]
        for i in range(3):
            vertex = triangle[i]
            if signed_distance(plane, vertex, plane_point) < 0: num_out += 1
            else: in_bounds[i] = vertex
        if num_out == 0: return [triangle]
        
        # if one point is OOB, then make 2 new triangles
        elif num_out == 1:
            new_points = []
            for i in range(3):
                if in_bounds[i] is not None:
                    new_points.append(in_bounds[i])
                else:
                    new_points.append(line_plane_intersect(plane, triangle[(i-1)%3], triangle[i], plane_point))
                    new_points.append(line_plane_intersect(plane, triangle[(i+1)%3], triangle[i], plane_point))
                    
            triangle1 = np.array([new_points[0],new_points[1],new_points[2],triangle[-1]], dtype=object)
            triangle2 = np.array([new_points[0],new_points[2],new_points[3],triangle[-1]], dtype=object)
            return [triangle1, triangle2]
        
        elif num_out == 2:
            #new_vertices = vertices.copy()
            for i in range(3):
                if in_bounds[i] is not None:
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

"""class Transformed_List(list):
    def __new__(cls, iterable, camera):
        #transformed_iterable = [camera.transform_point(point) for point in iterable]
        return super().__new__(cls)
    def __init__(self, iterable, camera):
        transformed_iterable = [camera.transform_point(point) for point in iterable]
        super().__init__(transformed_iterable)


class Projected_Mesh(list):
    def __new__(cls, iterable, camera):
        return super().__new__(cls)
    def __init__(self, iterable, camera):
        transformed_iterable = [[to_pygame(camera.perspective_projection(point)) for point in polygon[:-1]] for polygon in iterable]
        super().__init__(transformed_iterable)
"""

class Vertex():
    def __init__(self, x=0,y=0,z=0,w=1, color = (255,255,255)) -> None:
        self.hcoords = np.array([x,y,z,w])
        self.color = color


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running: bool = True
    clock = pygame.time.Clock() 
    cam = Camera()
    cam.third_person = False
    indices, points, maxval, normals, all_indices = read_obj('obj_files/mountains.obj')
    position = np.array([0,0,0,0])
    object = Object(points,position,indices,cam,screen,normals,all_indices,maxval)
    position = np.array([-260,0,1000,0])
    #object2 = Object(points,position,indices,cam,screen,maxval)
    angle=math.pi/180
    held: bool = False
    #object.draw()
    rate = 10*60
    
    for _ in range(100):
        cam.rotate_cam(np.array([1,0,0]),angle)
        cam.move_cam(np.array([0,0,10,0]))
        screen.fill("slate gray")
        object.draw3()
        pygame.display.flip()
    
    """while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                held = True
            elif event.type == pygame.MOUSEBUTTONUP:
                held = False
            if event.type == pygame.MOUSEMOTION and held:
                dx, dy = event.rel
                # Rotate about y axis when mouse moves L->R and x-axis when mouse moves UP->DOWN
                cam.rotate_cam(np.array([1,0,0]),angle*dy/10)
                cam.rotate_cam(np.array([0,1,0]),-angle*dx/10)

        # CONTROLS
        keys = pygame.key.get_pressed()
        # if want simultaneous movement change these all to ifs
        fps = clock.get_fps()
        if fps == 0:
            fps = 30
        if keys[pygame.K_w]:
            cam.move_cam(np.array([0,0,rate/fps,0]))
        elif keys[pygame.K_s]:
            cam.move_cam(np.array([0,0,-rate/fps,0]))
        if keys[pygame.K_d]:
            cam.move_cam(np.array([rate/fps,0,0,0]))
        elif keys[pygame.K_a]:
            cam.move_cam(np.array([-rate/fps,0,0,0]))
        if keys[pygame.K_SPACE]:
            cam.move_cam(np.array([0,-rate/fps,0,0]))
        elif keys[pygame.K_LSHIFT]:
            cam.move_cam(np.array([0,rate/fps,0,0]))  
        elif keys[pygame.K_t]:
            cam.third_person = not cam.third_person
            print("POV", cam.third_person)
        if keys[pygame.K_UP]:
            cam.fovx += math.pi/180
            cam.update()
        elif keys[pygame.K_DOWN]:
            cam.fovx -= math.pi/180
            cam.update()

        # DRAWING SCREEN
        # TODO maybe add a more efficient method to just draw over previous lines
        screen.fill("slate gray")
        object.draw3()
        #object2.draw()

        pygame.display.flip()
        clock.tick(60)
        print(clock)"""
        


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)

    #main()


pygame.quit()