import numpy as np
import math
from RendererGit import *
#from rayMarching3D import *
from constants import *
import random

from terraingen import *

class Box(Object):
    def __init__(self, dimensions, points, center, indices, camera, screen, normals, all_indices, maxval=1, tofrust = True) -> None:
        self.center = center
        self.dimensions = dimensions/2
        # remove center later
        Object.__init__(self, points, center, indices, camera, screen, normals, all_indices, maxval=maxval, tofrust=tofrust)

    def dist_func(self, pos) -> float:
        offset = np.abs(pos - self.center) - self.dimensions
        unsigneddist = length3D(np.maximum(offset, np.array([0,0,0,0])),[0,0,0,0])
        distinsidebox = max(np.minimum(offset, np.array([0,0,0,0])))
        return unsigneddist+distinsidebox
    # add derivatives of sdf for gradient vector
    
class Sphere:
    def __init__(self, center, radius) -> None:
        self.center = center
        self.radius = radius
    def dist_func(self, pos) -> float:
        return length3D(pos,self.center) - self.radius

class Triangle:
    def __init__(self, vertices, normal, center) -> None:
        self.vertices = vertices
        self.normal = normal
        self.center = center
    def dist_func(self, pos) -> float:
        return

class Ellipsoid:
    def __init__(self, center, radii) -> None:
        self.center = center
        self.radii = radii
    def dist_func(self, pos) -> float:
        return

def length3D(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

def get_box_dimensions(vertices):
    x_values = [vertex[0] for vertex in vertices]
    y_values = [vertex[1] for vertex in vertices]
    z_values = [vertex[2] for vertex in vertices]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    min_z, max_z = min(z_values), max(z_values)
    width = max_x - min_x
    height = max_y - min_y
    depth = max_z - min_z
    return np.array([width, height, depth,0])



def main():
    cam = Camera()
    clock = pygame.time.Clock() 
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    running = True
    held = False
    indices, points, maxval, normals, all_indices = read_obj('obj_files/cube.obj')
    position = np.array([0,0,0,0])
    object = Object(points,position,indices,cam,screen,normals,all_indices,maxval)
    position = np.array([0,0,0,0])
    dimensions = get_box_dimensions(points)
    #box1 = Box(dimensions,points,position,indices,cam,screen,maxval)
    angle=math.pi/180
    shapes = []
    box_indices = indices
    box_points = points

    grid_size = 128
    noise_map = generate_noise(grid_size) * 8 
    noise_map = noise_map - np.max(noise_map)/2
    
    map_size = grid_size/16
    for i in range(int(map_size)):
        for j in range(int(map_size)):
        # Add random rotation using quaternion
        #position = np.array([random.uniform(-maxval,maxval),random.uniform(-maxval,maxval),random.uniform(maxval,2*maxval),0])
        #position = np.array([random.randint(-10,10),random.randint(-10,10),random.randint(10,2*10),0])
            #print(int(noise_map[i][j]))
            position = np.array([i-map_size/2,j-map_size/2,noise_map[i][j],0])
            
            points2 = np.array([p + position for p in box_points])
            #shapes.append(Box(dimensions,points,position,indices,cam,screen,maxval))
            #points += points2
            
            points = np.concatenate((points, points2))
            box_indices = [index + len(box_points) for index in box_indices]
            indices = np.concatenate((indices, box_indices))
        
        #indices += indices
    shapes.append(Box(dimensions,points,position,indices,cam,screen,normals,all_indices,maxval))

    cam.shapes = shapes
    rate = 10*60
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
                cam.rotate_cam(np.array([1,0,0]),angle*dy/10)
                cam.rotate_cam(np.array([0,1,0]),-angle*dx/10)
                # TODO make this dependent on fps?
        


        # CONTROLS
        keys = pygame.key.get_pressed()
        # if want simultaneous movement change these all to ifs
        fps = clock.get_fps()
        if fps == 0:
            fps = 30
        if keys[pygame.K_w]:
            cam.move_cam(np.array([0,0,rate/fps,0]),True)
        elif keys[pygame.K_s]:
            cam.move_cam(np.array([0,0,-rate/fps,0]),True)
        if keys[pygame.K_d]:
            cam.move_cam(np.array([rate/fps,0,0,0]),True)
        elif keys[pygame.K_a]:
            cam.move_cam(np.array([-rate/fps,0,0,0]),True)
        if keys[pygame.K_SPACE]:
            cam.move_cam(np.array([0,-rate/fps,0,0]),True)
        elif keys[pygame.K_LSHIFT]:
            cam.move_cam(np.array([0,rate/fps,0,0]),True)  
        elif keys[pygame.K_t]:
            cam.third_person = not cam.third_person
            print("POV", cam.third_person)
        if keys[pygame.K_UP]:
            cam.fovx += (math.pi/180*rate/(10*fps))
            cam.update()
        elif keys[pygame.K_DOWN]:
            cam.fovx -= (math.pi/180*rate/(10*fps))
            cam.update()
 
        # DRAWING SCREEN
        screen.fill("slate gray")

        #cam.raymarch()
        #object.draw()
        for shape in shapes:
            shape.draw()
        #box1.draw()

        pygame.display.flip()
        clock.tick(60)
        print(clock)
        


if __name__ == '__main__':
    main()

pygame.quit()