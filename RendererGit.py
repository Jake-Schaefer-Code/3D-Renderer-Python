import numpy as np
import math
import pygame
import random
from memory_profiler import profile
from boids import Boid
import time
from collections import deque

from quaternions import q_rot,q_mat_rot
from objreader import read_obj
from functions import *
from constants import *
from rayMarching3D import *
#from Objects import *   

# more .obj files https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html

# TODO
    # Create index and staging buffer data structures maybe?
    # UV coordinates for textures - divide them by position rather than have them be their own entity
    # Terrain:
        # Render terrain in 8x8 squares where each square is 2 triangles
        # Triangle strips only 3 vertices for first triangle, but every vertex after that builds a new triangle off the last
# TODO add a swtich between first person and 3rd person around perspective center of the screen (0,0,0,0)

# Camera is at (0,0) and is always looking down the -z axis
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
        point2 = self.matrix @ point
        return point2
    
    # Projecting point onto projection plane
    def perspective_projection(self, point):
        persp_point = self.perspM @ point
        if persp_point[3] == 0:
            return persp_point/0.000001
        else: 
            return persp_point/persp_point[3]

    # Camera movement
    def move_cam(self, trans_vector, key_move=False) -> None:
        if key_move:
            self.center3 = translate(np.identity(4), trans_vector) @ self.center3
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
    
    # For eventual raymarching
    def distance_to_nearest(self, pos) -> float:
        nearest_dist = 1000000
        for shape in self.shapes:
            shape_dist = shape.dist_func(pos)
            nearest_dist = min(nearest_dist, shape_dist)
        return nearest_dist
    
    # Eventual raymarching shaders
    def raymarch(self):
        self.dist_to_nearest = self.distance_to_nearest(self.pos)
        z = self.dist_to_nearest
        tmp_near = self.dist_to_nearest
        tmp_pos = self.pos + tmp_near * np.array([-WIDTH/2, -HEIGHT/2, 1,0]) / length3D([-WIDTH/2, -HEIGHT/2, 1,0], [0,0,0])
        for x_step in range(STEPS):
            x = x_step * WIDTH/STEPS - WIDTH/2
            for y_step in range(STEPS):
                y = y_step * HEIGHT/STEPS - HEIGHT/2
                angle_vector = np.array([x,y,z,0]) / length3D([x,y,z,0],[0,0,0])
                tmp_pos = self.pos + tmp_near * angle_vector
                tmp_near = self.distance_to_nearest(tmp_pos)
                while tmp_near > 10 and tmp_near < 3600:
                    tmp_near = self.distance_to_nearest(tmp_pos)
                    new_pos = tmp_pos + tmp_near * angle_vector
                    tmp_pos = new_pos
                
                

# TODO make sure separate objects are drawn in order of their distance to camera
class Object:
    def __init__(self, points, position, indices, camera, screen, maxval=1, tofrust = True):
        self.cam = camera
        # Local Origin
        self.screen = screen
        self.maxval=maxval
        
        if tofrust:
            self.points = np.array([self.to_frustum(p) for p in points])

        else:
            #self.points = np.array([p/self.maxval for p in points])
            self.points = np.array([p for p in points])

        self.points = np.array([p+position for p in self.points])
        
        self.position = position
        self.center = position


        self.transformpoints = points
        self.todraw = np.array([])

        self.numpoints = len(points)
        self.polygons = indices
        self.numfaces = len(self.polygons)
        self.lightdir = np.array([0,0,-1,0])
        
        self.planes = PLANES
        self.plane_points = PLANE_POINTS

    # converts to frustum coordinates
    def to_frustum(self,point):
        z = NEAR_Z*((point[2])/self.maxval) + 2* self.cam.center[2]
        x = NEAR_Z*((point[0])/self.maxval)
        y = -NEAR_Z*((point[1])/self.maxval)
        return np.array([x,y,z,point[3]])

    def draw(self) -> None:

        for i in range(self.numpoints):
            self.transformpoints[i] = self.cam.transform_point(self.points[i])

        
        self.lightdir = self.cam.transform_point(np.array([0,0,-1,0]))
        #self.center = self.cam.transform_point(self.position)
        todraw = []
        for i in range(self.numfaces):
            indices = self.polygons[i]
            t = [self.transformpoints[i] for i in indices]+[0.0]
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
        drawmesh = self.zordermesh(todraw)
        for i in range(len(drawmesh)):
            t = drawmesh[i][:-1]
            c = drawmesh[i][-1]

            # Projects points and converts them to pygame coordinates
            t = [self.cam.perspective_projection(p) for p in t]
            t = [to_pygame(p) for p in t]

            # Draws Triangles
            pygame.draw.polygon(self.screen,(185*c,245*c,185*c), t, width = 0)
            # Draws edges on triangles
            #pygame.draw.polygon(self.screen,"white", t, width = 1)

    def zordermesh(self, mesh):
        mesh3 = []
        for i in range(len(mesh)):
            mesh3.append([m[2] for m in mesh[i][:-1]])
        if mesh3 == []:
            return mesh
        means = np.mean(mesh3, axis=1)
        reorder = np.argsort(means)[::-1]
        new_mesh = []
        for i in range(len(mesh)):
            new_mesh.append(mesh[reorder[i]])
        return new_mesh

    def clip_mesh(self, t):
        # TODO Something is wrong with clipping on top and bottom planes
        triangle_queue = deque(t)      
        for i in range(len(self.cam.planes)):
            plane = self.cam.planes[i]
            plane_point = np.array([0,0,0,0])
            for _ in range(len(triangle_queue)):
                triangle = triangle_queue.popleft()
                new_triangles = self.clip_triangle(plane, plane_point, triangle)
                if new_triangles is not None: 
                    
                    triangle_queue.extend(new_triangles)
        return list(triangle_queue)

    def clip_triangle(self, plane, plane_point, triangle):
        # plane equation: a(x-xo)+b(y-yo)+c(z-zo)=0
        in_bounds = [None,None,None]
        num_out = 0
        vertices = triangle[:-1]
        
        for i in range(3):
            vertex = triangle[i]
            #print(vertex[1])
            #print()
            # TODO change perhaps
            if signed_distance(plane, plane_point, vertex) < 0:
                num_out += 1
                #print(vertex)
                #print(vertex, plane)
            else:
                in_bounds[i] = vertex
                
        if num_out == 0:
            return [triangle]
        
        # if one point is OOB, then make 2 new triangles
        elif num_out == 1:
            new_points = []
            for i in range(3):
                if in_bounds[i] is not None:
                    new_points.append(in_bounds[i])
                else:
                    
                    p4 = line_plane_intersect(plane, plane_point, vertices[(i-1)%3], triangle[i])
                    new_points.append(p4)
                    p5 = line_plane_intersect(plane, plane_point, vertices[(i+1)%3], triangle[i])
                    new_points.append(p5)
                    

            triangle1 = np.array([new_points[0],new_points[1],new_points[2],triangle[-1]], dtype=object)
            triangle2 = np.array([new_points[0],new_points[2],new_points[3],triangle[-1]], dtype=object)
            return [triangle1, triangle2]
        
        elif num_out == 2:
            new_vertices = vertices.copy()
            for i in range(3):
                if in_bounds[i] is not None:
                    # intersection of plane and line from current vertex to both OOB vertices
                    p4 = line_plane_intersect(plane, plane_point, vertices[(i+1)%3], triangle[i])

                    new_vertices[(i+1)%3] = p4
                    new_vertices[(i-1)%3] = line_plane_intersect(plane, plane_point, vertices[(i-1)%3], triangle[i])
                    new_triangle = np.array([new_vertices[0], new_vertices[1], new_vertices[2], triangle[-1]], dtype=object)
            return [new_triangle]
        
        elif num_out == 3:
            return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running = True
    clock = pygame.time.Clock() 
    cam = Camera()
    cam.third_person = False
    indices, points, maxval, normals = read_obj('obj_files/cube.obj')
    position = np.array([0,0,0,0])
    object = Object(points,position,indices,cam,screen,maxval)
    position = np.array([-260,0,1000,0])
    object2 = Object(points,position,indices,cam,screen,maxval)
    angle=math.pi/180
    axis = np.array([0,1,0])
    held = False
    point = to_pygame(cam.perspective_projection(np.array([1500,0,1000,1])))
    print(point)
    print(signed_distance(PLANES[0],np.array([0,0,0,0]), np.array([1500,0,1000,1])))
    print(signed_distance(PLANES[3],np.array([0,0,0,0]), np.array([350.3,-1144.47,1144.306,1])))
    print(to_pygame(cam.perspective_projection(np.array([350.3,-1144.47,1144.306,1]))))
    
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                held = True
                """if event.button == 4:  # Scroll up
                    cam.move_cam(np.array([0,0,-10,0]))
                elif event.button == 5:  # Scroll down
                    cam.move_cam(np.array([0,0,10,0]))"""
            elif event.type == pygame.MOUSEBUTTONUP:
                held = False
            if event.type == pygame.MOUSEMOTION and held:
                dx, dy = event.rel
                # Rotate about y axis when mouse moves L->R and x-axis when mouse moves UP->DOWN
                cam.rotate_cam(np.array([1,0,0]),-angle*dy/10)
                cam.rotate_cam(np.array([0,1,0]),angle*dx/10)


        # CONTROLS
        keys = pygame.key.get_pressed()
        # if want simultaneous movement change these all to ifs
        if keys[pygame.K_w]:
            cam.move_cam(np.array([0,0,10,0]),True)
        elif keys[pygame.K_s]:
            cam.move_cam(np.array([0,0,-10,0]),True)
        elif keys[pygame.K_d]:
            cam.move_cam(np.array([10,0,0,0]),True)
        elif keys[pygame.K_a]:
            cam.move_cam(np.array([-10,0,0,0]),True)
        elif keys[pygame.K_SPACE]:
            cam.move_cam(np.array([0,-10,0,0]),True)
        elif keys[pygame.K_LSHIFT]:
            cam.move_cam(np.array([0,10,0,0]),True)
        elif keys[pygame.K_t]:
            cam.third_person = not cam.third_person
        elif keys[pygame.K_UP]:
            cam.fovx += math.pi/180
            cam.update()
        elif keys[pygame.K_DOWN]:
            cam.fovx -= math.pi/180
            cam.update()

        # DRAWING SCREEN
        # TODO maybe add a more efficient method to just draw over previous lines
        screen.fill("slate gray")
        object.draw()
        #object2.draw()

        pygame.display.flip()
        clock.tick(60)
        #print(clock)
        


if __name__ == '__main__':
    main()

pygame.quit()