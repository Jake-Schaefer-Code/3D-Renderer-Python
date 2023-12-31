import numpy as np
import math
import pygame
from quaternions import q_rot,q_mat_rot
from objreader import read_obj
import random
from memory_profiler import profile
from boids import Boid
import time

# more .obj files https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html

# TODO
    # Create index and staging buffer data structures maybe?
    # UV coordinates for textures - divide them by position rather than have them be their own entity
    # Terrain:
        # Render terrain in 8x8 squares where each square is 2 triangles
        # Triangle strips only 3 vertices for first triangle, but every vertex after that builds a new triangle off the last

WIDTH = 720
HEIGHT = 720
#WIDTH = 1400
#HEIGHT = 1080
ASPECT_RATIO = HEIGHT/WIDTH
FOVX = math.pi/2    
FOVY = 2 * math.atan(ASPECT_RATIO * math.tan(FOVX/2))

# define coordinate origin to be in center of the screen, input point is in this form
SCREEN_ORIGIN = np.array([WIDTH/2, HEIGHT/2,0,1])
SCREEN_SCALE = np.array([WIDTH/2, HEIGHT/2,1,0])

center = np.array([0,0,654.545])

# Camera is at (0,0) and is always looking down the -z axis
class Camera:
    def __init__(self):
        self.dir = np.array([0,0,-1,0],dtype='float64')
        self.pos = np.array([0,0,0,1],dtype='float64')
        self.lightdir = np.array([1,1,1,0],dtype='float64')
        
        # Distance (scalar) to near (projection) plane
        self.nearz = 360
        self.lbn = self.nearz * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), -1.0])

        # Distance (scalar) to far plane
        self.farz = 3600
        self.rtf = self.nearz * np.array([math.tan(FOVX/2), math.tan(FOVY/2), -self.farz/self.nearz])
        
        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = self.to_frustum(np.array([0,0,0,1]))
        
        # The matrix that will be modified through transitions and rotations
        self.matrix = np.identity(4)

        l,b,n = self.lbn
        r,t,f = self.rtf
        self.perspM = np.array([[2*n/(r-l),0,0,-n*(r+l)/(r-l)],
                        [0,2*n/(t-b),0,-n*(t+b)/(t-b)],
                        [0,0,-(n+f)/(f-n),2*n*f/(n-f)],
                        [0,0,-1,0]])

    def perspective_projection(self,point):
        # Uncomment later if actually updating any of these at any point, but shouldn't be as of my current knowledge
        """l,b,n = self.nearz * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), -1.0])
        r,t,f = self.nearz * np.array([math.tan(FOVX/2), math.tan(FOVY/2), -self.farz/self.nearz])
        self.perspM = np.array([[2*n/(r-l),0,0,-n*(r+l)/(r-l)],
                        [0,2*n/(t-b),0,-n*(t+b)/(t-b)],
                        [0,0,-(n+f)/(f-n),2*n*f/(n-f)],
                        [0,0,-1,0]])"""
        
        point2 = self.perspM @ self.matrix @ point
        return point2/point2[3]

    def transform_point(self, point):
        point2 = self.matrix @ point
        return point2
    
    def project_point(self, point):
        point2 = self.perspM @ point
        return point2/point2[3]
    """def update_cam(self):
        self.lbn = self.nearz * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), -1.0])
        self.rtf = self.nearz * np.array([math.tan(FOVX/2), math.tan(FOVY/2), -self.farz/self.nearz])"""

    # maybe move this to object class
    def move_cam(self, trans):
        self.pos -= trans
        tx, ty, tz, w = trans
        trans_matrix = np.array([[1,0,0,tx],
                                 [0,1,0,ty],
                                 [0,0,1,tz],
                                 [0,0,0,1 ]]) 
        
        self.matrix = trans_matrix @ self.matrix

    def rotate_cam(self, axis, angle):
        self.matrix = q_mat_rot(self.matrix, axis, angle)
    
    # converts to canonical viewing space
    def to_frustum(self,point):
        n = self.nearz
        f = self.farz
        c1 = 2*f*n/(n-f)
        c2 = (f+n)/(f-n)
        z = c1/(point[2]-c2)
        x = point[0]
        y = -point[1]
        return np.array([x,y,z,point[3]])
    

class Triangle:
    def __init__(self, indices, color) -> None:
        self.indices = indices
        self.color = color

# TODO make sure separate objects are drawn in order of their distance to camera
class Object:
    def __init__(self, center, points, indices, camera, screen, maxval=1, tofrust = True):
        self.cam = camera
        # Local Origin
        self.center = center
        self.screen = screen
        self.maxval=maxval
        
        if tofrust:
            self.points = np.array([self.to_frustum(p) for p in points])
        else:
            self.points = points

        #self.projpoints = points

        self.transformpoints = points
        self.todraw = np.array([])

        self.numpoints = len(points)
        self.polygons = indices
        self.numfaces = len(self.polygons)

        self.lightdir = np.array([0,0,-1,0])

        # TODO maybe combine colors and polygons into one array? Or is it more beneficial to 
        # reference this array using indices
        
        self.colorvals = np.array([(0,0,0) for _ in range(self.numfaces)])
        self.colors = np.array([(i*255/self.numfaces,0,0) for i in range(self.numfaces)])
        #self.zorder()
        #self.reorder = enumerate(indices)
    
    # converts to frustum coordinates
    def to_frustum(self,point):
        x = 360*(point[0]/self.maxval)
        y = -360*(point[1]/self.maxval)
        # change to centerval
        z = 360*(point[2]/self.maxval) + 360 + self.cam.center[2]
        return np.array([x,y,z,point[3]])
    
    def get_face_normal(self, face):
        # TODO change this so that if there is an option such that they face away or toward the camera, they face toward
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

    def draw(self):
        for i in range(self.numpoints):
            # applies transformations to points
            self.transformpoints[i] = self.cam.transform_point(self.points[i])

        self.lightdir = self.cam.transform_point(np.array([0,0,-1,0]))

        todraw = []
        colorvals = []
        for i in range(self.numfaces):
            indices = self.polygons[i]
            t = self.transformpoints[indices]
            normal = self.get_face_normal(t)
            #face_center = np.mean(t,axis=0)
            # only draw if face is facing camera i.e. if normal is facing in negative direction
            coincide = np.dot(normal, t[1] - np.array([0,0,0,0]))
            if coincide < 0:
                colorval = np.abs((normal[0]*self.lightdir[0] + 
                                   normal[1]*self.lightdir[1] + 
                                   normal[2]*self.lightdir[2])/2)

                todraw.append(t)
                colorvals.append(colorval)
                #pygame.draw.polygon(self.screen,(185*colorval,245*colorval,185*colorval), t, width=0)
                # draws normals of each polygon for test
                #pygame.draw.circle(self.screen,"red", to_pygame(face_center), 2)
                #pygame.draw.line(self.screen, "white", to_pygame(face_center), to_pygame(face_center + np.append(normal,1)))
        drawmesh, drawcolors = self.zordermesh(todraw, colorvals)
        for i in range(len(drawmesh)):
            t = drawmesh[i]
            c = drawcolors[i]
            # TODO can make this more efficient?
            # projects points and converts them to pygame coordinates
            t = [self.cam.project_point(p) for p in t]
            t = [to_pygame(p) for p in t]
            pygame.draw.polygon(self.screen,(185*c,245*c,185*c), t, width = 0)
            #pygame.draw.polygon(self.screen,"black", t, width = 1)

    def zordermesh(self, mesh, colors):
        reorder = np.argsort(np.mean(mesh, axis=1)[:,2])[::-1]
        return np.array(mesh)[reorder], np.array(colors)[reorder]


class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir
    def draw(self, screen):
        pygame.draw.line(screen, "white", to_pygame(self.origin), to_pygame(self.origin - self.dir))

class Hit:
    def __init__(self, object):
            self.object = object
    def hitTriangle(self, ray, triangle, normal):
        ao = ray.origin - triangle[0]
        dao = np.cross(ao, ray.dir)
        determinant = -np.dot(ray.dir,normal)

        dist = np.dot(ao, normal)/determinant




def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def to_pygame(point):
    return ((point * SCREEN_SCALE) + SCREEN_ORIGIN)[:-2]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running = True
    clock = pygame.time.Clock() 
    cam = Camera()
    
    boids = np.array([Boid(np.array([np.random.random()*10,np.random.random()*10,np.random.random()*10,1])) for _ in range(10)])
    """for _ in range(10):
        loc = np.array([np.random.random()*10,np.random.random()*10,np.random.random()*10,1])
        boid = Boid(loc)
        np.append(boids, boid)"""

    
    #objects = np.array([Object(center, boid.vertices, boid.faces, cam, screen, boid.maxval) for boid in boids])


    boid = Boid()
    indices, points, maxval = boid.faces, boid.vertices, boid.maxval

    indices, points, maxval = read_obj('obj_files/mountains.obj')
    object = Object(center,points,indices,cam,screen,maxval)

    circlepoint = np.array([0,0,360,1])
    circlepoint = cam.perspective_projection(circlepoint)

    angle=math.pi/180
    axis = np.array([0,1,0])
    circlepoint = q_rot(circlepoint,axis,angle)

    held = False
    
    numx = 16
    numy = int(numx*ASPECT_RATIO)
    #ray1 = Ray(np.array([0,0,0,1]),np.array([1,1,1,0]))
    #rays = np.array([Ray(cam_origin, np.array([(2*i)/numx-1,(2*j)/numy-1,1,0])) for i in range(numx+1) for j in range(numy+1)])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                held = True
                if event.button == 4:  # Scroll up
                    cam.move_cam(np.array([0,0,-10,0]))
                elif event.button == 5:  # Scroll down
                    cam.move_cam(np.array([0,0,10,0]))
            elif event.type == pygame.MOUSEBUTTONUP:
                held = False
            if event.type == pygame.MOUSEMOTION and held:
                dx, dy = event.rel
                # Rotate about y axis when mouse moves L->R and x-axis when mouse moves UP->DOWN
                cam.rotate_cam(np.array([1,0,0]),angle*dy/10)
                cam.rotate_cam(np.array([0,1,0]),-angle*dx/10)
                # TODO make this dependent on fps?


        # CONTROLS
        keys = pygame.key.get_pressed()
        # if want simultaneous movement change these all to ifs
        if keys[pygame.K_w]:
            cam.move_cam(np.array([0,0,-10,0]))
        elif keys[pygame.K_s]:
            cam.move_cam(np.array([0,0,10,0]))
        elif keys[pygame.K_d]:
            cam.move_cam(np.array([-10,0,0,0]))
        elif keys[pygame.K_a]:
            cam.move_cam(np.array([10,0,0,0]))
        elif keys[pygame.K_SPACE]:
            cam.move_cam(np.array([0,10,0,0]))
        elif keys[pygame.K_LSHIFT]:
            cam.move_cam(np.array([0,-10,0,0]))
 

        # DRAWING SCREEN
        # TODO maybe add a more efficient method to just draw over previous lines
        screen.fill("slate gray")


        """for obj in objects:
            obj.draw()"""

        object.draw()

        #p1 = to_pygame(cam.perspective_projection(cam.pos))
        #p2 = to_pygame(np.array([0,0,0,1]) + cam.dir)
        #p2 = to_pygame(np.array([1,1,1,1])*-cam.perspective_projection(np.array([0,0,3600,1])))
        #pygame.draw.line(screen, "white", (360,360), p2[:-2])


        """for ray in rays:
            ray.draw(screen)"""

        pygame.display.flip()
        clock.tick(60)
        print(clock)
        


if __name__ == '__main__':
    main()

pygame.quit()