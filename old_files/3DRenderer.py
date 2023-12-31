import numpy as np
import math
import pygame
from quaternions import q_rot,q_mat_rot
from objreader import read_obj
import random
import boids
# more .obj files https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html

# TODO
    # Create index and staging buffer data structures maybe?


WIDTH = 720
HEIGHT = 720
#WIDTH = 1400
#HEIGHT = 1080
ASPECT_RATIO = HEIGHT/WIDTH
FOVX = math.pi/2
FOVY = math.pi/2

# fov scaling factor f
FX_SCALE = 1/math.tan(FOVX/2)
FY_SCALE = 1/math.tan(FOVY/2)

origin = np.array([0,0,654.545])
center = origin
"""size = 200
p1 = np.array([center[0] - size, center[1] - size, center[2] + size,1])
p2 = np.array([center[0] - size, center[1] + size, center[2] + size,1])
p3 = np.array([center[0] + size, center[1] + size, center[2] + size,1])
p4 = np.array([center[0] + size, center[1] - size, center[2] + size,1])
p5 = np.array([center[0] - size, center[1] - size, center[2] - size,1])
p6 = np.array([center[0] - size, center[1] + size, center[2] - size,1])
p7 = np.array([center[0] + size, center[1] + size, center[2] - size,1])
p8 = np.array([center[0] + size, center[1] - size, center[2] - size,1])
pointlist = [p1,p2,p3,p4,p5,p6,p7,p8]
#lineList = [(p1,p2),(p2,p3),(p3,p4),(p4,p1), (p5,p6),(p6,p7),(p7,p8),(p8,p5), (p5,p1),(p6,p2),(p7,p3),(p8,p4),(p1,p3)]
lineList = [(p1,p3),(p1,p4),(p3,p4),(p3,p7),(p3,p8),(p7,p8)]
triangleList = [(p1,p2,p3),(p1,p3,p4),(p3,p7,p8), (p4,p3,p8),(p7,p8,p5),(p5,p6,p7),(p1,p2,p6),(p5,p6,p1)]

x_ax = [np.array([0,0,654.545,1]),np.array([360,0,654.545,1])]
y_ax = [np.array([0,-360,654.545,1]),np.array([0,0,654.545,1])]
z_ax = [np.array([0,0,654.545,1]),np.array([0,0,654.545+360,1])]"""

# Camera is at (0,0) and is always looking down the -z axis
class Camera:
    def __init__(self, origin=np.array([WIDTH/2,HEIGHT/2, -360, 1]), orient=np.array([0,0,0],dtype='float64')):

        self.dir = np.array([0,0,1,1],dtype='float64')
        self.pos = np.array([0,0,0],dtype='float64')

        self.lightdir = np.array([1,-1,1,1],dtype='float64')
        
        # Distance (scalar) to near (projection) plane
        self.nearz = 360
        self.lbn = self.nearz * np.array([-math.tan(FOVX/2), -math.tan(FOVY/2), -1.0])

        # Distance (scalar) to far plane
        self.farz = 3600
        self.rtf = self.nearz * np.array([math.tan(FOVX/2), math.tan(FOVY/2), -self.farz/self.nearz])
        
        # This is where [0,0,0] is in the canonical viewing space and thus the center of rotation
        self.center = -1 * self.to_frustum(np.array([0,0,0,1]))[:-1]

        self.matrix = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        
        self.idmatrix = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0],
                                [0,0,0,1]])
        
        # camera orientation
        self.camera_angle = np.array([0,0,0],dtype='float64')
        self.near_vec = np.array([0,0,360,1])
        self.far_vec = np.array([0,0,3600,1])


        self.axis = np.array([1,1,1])
        # maybe to get axis, take angle and multiply it by axis, so if only rotating in x dir, y and z axes are zero
        #self.rot_axis = self.axis.T @ self.camera_angle
        self.rot_angle = math.sqrt(np.sum(np.dot(self.camera_angle, self.camera_angle)))

        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])

        #self.rotx = q_mat_rot(self.matrix, np.array([1,0,0]), self.camera_angle[0])
        #self.roty = q_mat_rot(self.matrix, np.array([1,0,0]), self.camera_angle[0])
        #self.rotz = q_mat_rot(self.matrix, np.array([1,0,0]), self.camera_angle[0])

    def update_cam(self):
        self.rot_axis = self.camera_angle
        mag_vec = np.array([n*n for n in self.camera_angle])
        self.rot_angle = math.sqrt(np.sum(mag_vec))

    def perspective_projection(self,point):
        l,b,n = self.lbn
        r,t,f = self.rtf
        perspM = np.array([[2*n/(r-l),0,0,-n*(r+l)/(r-l)],
                        [0,2*n/(t-b),0,-n*(t+b)/(t-b)],
                        [0,0,-(n+f)/(f-n),2*n*f/(n-f)],
                        [0,0,-1,0]])
        point2 = perspM @ point
        return point2/point2[3]

    def move_cam(self, trans):
        tx, ty, tz = trans
        self.pos += trans
        trans_matrix = np.array([[1,0,0,tx],
                                 [0,1,0,ty],
                                 [0,0,1,tz],
                                 [0,0,0,1 ]]) 

        self.matrix = trans_matrix @ self.matrix


    def rotate_cam(self, axis, angle):
        dist_to_center = self.center - self.pos

        # should just be rotation of object
        self.move_cam(dist_to_center)
        self.matrix = q_mat_rot(self.matrix, axis, angle/2)
        self.move_cam(-dist_to_center)

        # for first person:
        self.move_cam(-dist_to_center)
        self.matrix = q_mat_rot(self.matrix, axis, angle/2)
        self.move_cam(dist_to_center)


        self.dir = q_rot(self.dir,axis,angle)
        
    def to_frustum(self,point):
        n = self.nearz
        f = self.farz
        c1 = 2*f*n/(n-f)
        c2 = (f+n)/(f-n)
        z = c1/(point[2]-c2)
        x = point[0]
        y = -point[1]
        return np.array([x,y,z,point[3]])

class Object:
    def __init__(self, center, points, triangles, color, camera, screen, maxval=1, tofrust = True):
        
        self.cam = camera
        # Local Origin
        self.center = center
        self.screen = screen
        self.color = color
        
        # TODO maybe try storing indices of points instead of a whole mesh of points
        self.points = points
        self.triangles = triangles
        self.tmptri = self.triangles.copy()
        self.lasttmptri = self.triangles.copy()
        self.colors = np.array([(0,0,0) for _ in range(len(self.triangles))])
        self.zorder()

        
        self.maxval=maxval
        if tofrust:
            for i in range(len(self.triangles)):
                t = self.triangles[i]
                self.triangles[i] = [self.to_frustum(t[j]) for j in range(len(t))]

        

    def to_frustum(self,point):
        x = 360*(point[0]/self.maxval)
        y = -360*(point[1]/self.maxval)
        z = 360*(point[2]/self.maxval) +360 + 560
        return np.array([x,y,z,point[3]])

    
    def draw(self):
        
        for i in range(len(self.triangles)):
            self.tmptri[i] = [self.cam.matrix @ self.triangles[i][j] for j in range(len(self.triangles[i]))]
        """if np.any(self.tmptri) != np.any(self.lasttmptri):
            self.zorder()
            self.lasttmptri = self.tmptri"""
        self.zorder()
        self.lasttmptri = self.tmptri

        

        for i in range(len(self.triangles)):
            
            t = self.tmptri[i]
            color = self.colors[i]
            if np.any(color>255):
                color = (0,0,0)

            t = np.array([self.cam.perspective_projection(t[i]) for i in range(len(t))])
            condition = (t > 2) | (t < -2)
            if not np.any(condition):
                #color = (int(np.abs(t[0][0]*255/2)),int(np.abs(t[0][1]*255/2)),int(np.abs(t[0][2]*255/2)))
                t = [to_pygame(t[i])[:-2] for i in range(len(t))]
                pygame.draw.polygon(self.screen,color, t, width=0)
                #pygame.draw.polygon(self.screen,(250,250,250), t, width=2)

        
        """for i in range(len(tmptri)):
            t = tmptri[i]
            t = [self.cam.matrix @ t[0],self.cam.matrix @ t[1],self.cam.matrix @ t[2]]
            tmptri[i] = np.array([self.cam.perspective_projection(t[0]),self.cam.perspective_projection(t[1]),self.cam.perspective_projection(t[2])])
        if np.any(tmptri) != np.any(self.triangles):
            self.zorder(tmptri)
        
        for i in self.indices:
            t = tmptri[i]
            condition = (t > 2) | (t < -2)
            if not np.any(condition):
                t = [to_pygame(t[0])[:-2],to_pygame(t[1])[:-2],to_pygame(t[2])[:-2]]
                #pygame.draw.polygon(self.screen,random_color(), t, width=5)
                pygame.draw.polygon(self.screen,"white", t, width=0)
                pygame.draw.polygon(self.screen,(250,250,250), t, width=2)"""
        
        """for point in self.points:
            point = self.cam.to_frustum(point)
            point = self.cam.matrix @ point
            point1 = self.cam.perspective_projection(point)
            if np.any(point1 > 2) or np.any(point1 < -2):
                pass
            else:
                point1 = to_pygame(point1)
                size = self.cam.to_frustum(point1)[2]
                pygame.draw.circle(self.screen,self.color,point1[:-2],5)"""
        
    def rotate(self,axis,angle):
        self.translate(-1*self.center)
        for i in range(len(self.triangles)):
            t = self.triangles[i]
            t = [q_rot(t[0],axis,angle),q_rot(t[1],axis,angle),q_rot(t[2],axis,angle)]
            self.triangles[i] = t
        for i in range(len(self.points)):
            point = self.points[i]
            point = q_rot(point,axis,angle)
            self.points[i] = point
        self.translate(self.center)

    def translate(self, trans):
        tx, ty, tz = trans
        trans_matrix = np.array([[1,0,0,tx],
                                 [0,1,0,ty],
                                 [0,0,1,tz],
                                 [0,0,0,1 ]]) 
        for i in range(len(self.triangles)):
            t = self.triangles[i]
            self.triangles[i] = [trans_matrix @ t[0], trans_matrix @ t[1], trans_matrix @ t[2]]
        for i in range(len(self.points)):
            self.points[i] = trans_matrix @ self.points[i]

    def zorder(self):
        # CHANGE??? 
        mesh = self.tmptri
        reorder = np.zeros(len(mesh))
        for i in range(len(mesh)):
            t = mesh[i]
            print(t)
            t = [[t[j][0]/360,-t[j][1]/360,(t[j][2]-920)/360,1] for j in range(len(t))]

            dir = t - self.cam.dir
            lightdir = t - self.cam.lightdir
            orderval = np.mean(np.linalg.norm(dir, axis=1))
            lightmean = np.mean(np.linalg.norm(lightdir, axis=1))
            #print(lightmean)
            self.colors[i] = (0,0,lightmean*255/4)
            reorder[i] = orderval
        self.indices = np.argsort(reorder)[::-1]
        self.tmptri = [mesh[i] for i in self.indices]



class Ray:
    def __init__(self):
        pass

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def to_pygame(point):
    # define coordinate origin to be in center of the screen, input point is in this form
    origin = np.array([WIDTH/2, HEIGHT/2,0,0])
    point = point*np.array([WIDTH/2, HEIGHT/2,1,1])
    return point + origin

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = np.array([n / mag for n in v])
    return v

def create_buffer(indices = np.array([],dtype="int32")):
    pass

def createVertexBuffer():
    pass

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running = True
    clock = pygame.time.Clock() 
    cam = Camera([360,360,0,1])
    """cube = Object(center,pointlist,triangleList,"white", cam, screen,tofrust=False)
    cube.draw()
    xaxis = Object(origin,[np.array([0,0,654.545,1])],[x_ax], "blue", cam,screen)
    axis = Object(origin,[np.array([0,0,654.545,1])],[y_ax], "red", cam,screen)
    zaxis = Object(origin,[np.array([0,0,654.545,1])],[z_ax], "green", cam,screen)"""

    mesh, points, maxval = read_obj('obj_files/cessna.obj')
    teapot = Object(center,points,mesh,"blue",cam,screen,maxval)
    #boid1 = boids.Boid()
    #mesh, points, maxval = boid1.faces, boid1.vertices, boid1.maxval
    #teapot = Object(center,points,mesh,"blue",cam,screen, maxval)

    circlepoint = np.array([0,0,360,1])
    circlepoint = cam.perspective_projection(circlepoint)

    angle=math.pi/180
    axis = np.array([0,1,0])
    circlepoint = q_rot(circlepoint,axis,angle)

    mouse_pressed = False
    boxsize = 160
    rotationbox = pygame.Rect(WIDTH-boxsize,0,boxsize,boxsize)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pressed = True
                if event.button == 4:  # Scroll up
                    cam.move_cam(np.array([0,0,-10]))
                elif event.button == 5:  # Scroll down
                    cam.move_cam(np.array([0,0,10]))
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = False
            elif event.type == pygame.MOUSEMOTION and mouse_pressed:
                dx, dy = event.rel
                #cam.camera_angle += np.array([dx/10*math.pi/180,dy/10*-math.pi/180,0])
                cam.rotate_cam(np.array([1,0,0]),math.pi/180*dy/10)
                cam.rotate_cam(np.array([0,1,0]),-math.pi/180*dx/10)

        # CONTROLS
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            cam.move_cam(np.array([0,0,-10]))
        elif keys[pygame.K_s]:
            cam.move_cam(np.array([0,0,10]))
        elif keys[pygame.K_d]:
            cam.move_cam(np.array([-10,0,0]))
        elif keys[pygame.K_a]:
            cam.move_cam(np.array([10,0,0]))
        elif keys[pygame.K_SPACE]:
            cam.move_cam(np.array([0,10,0]))
        elif keys[pygame.K_LSHIFT]:
            cam.move_cam(np.array([0,-10,0]))
        elif keys[pygame.K_r]:
            cam.rotate_cam(np.array([0,1,0]),math.pi/180)
        elif keys[pygame.K_e]:
            cam.rotate_cam(np.array([0,1,0]),-math.pi/180)

        # DRAWING SCREEN
        screen.fill("slate gray")
        pygame.draw.rect(screen,"white",rotationbox,5)
        teapot.draw()
        
        # DRAWING OBJETS
        #cube.rotate(axis,angle)
        #cube.draw()
        #xaxis.draw()
        #yaxis.draw()
        #zaxis.draw()

        """size = cam.to_frustum(circlepoint)[2]
        circlepoint = q_rot(circlepoint,axis,angle)
        pygame.draw.circle(screen,"white", to_pygame(circlepoint)[:-2],size/100)
        circlepoint = cam.to_frustum(circlepoint)
        circlepoint = cam.perspective_projection(circlepoint)
        for i in range(len(lineList)):
            line = lineList[i]
            #line = [cam.to_frustum(line[0]), cam.to_frustum(line[1])]
            line = [q_rot(line[0],axis,math.pi/180),q_rot(line[1],axis,math.pi/180)]
            line = [cam.perspective_projection(line[0]), cam.perspective_projection(line[1])]


            plotline = [cam.to_frustum(line[0])+360, cam.to_frustum(line[1])+360]
            #plotline = [line[0]+360, line[1]+360]
            size = cam.to_frustum(line[0])[2]



            pygame.draw.line(screen,"white",plotline[0][:-2],plotline[1][:-2])
            pygame.draw.circle(screen,"white",plotline[0][:-2],5)
            
            #line = [cam.to_frustum(line[0]),cam.to_frustum(line[1])]
            #lineList[i] = line
            #line = [cam.perspective_projection(line[0]),cam.perspective_projection(line[1])]
            lineList[i] = [cam.to_frustum(line[0]),cam.to_frustum(line[1])]
            # lineList[i] = [to_pygame(line[0]), to_pygame(line[1])]"""
        

        cam.update_cam()
        pygame.display.flip()
        clock.tick(60)
        #print(clock)
        
    

if __name__ == '__main__':
    main()

pygame.quit()