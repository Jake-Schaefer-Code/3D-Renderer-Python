import numpy as np
import math
import pygame as pg
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
from camera import *
from Object import *



def main(testing=False):
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    running: bool = True
    clock = pg.time.Clock() 
    cam = Camera()
    cam.third_person = False
    indices, points, maxval, normals, all_indices = read_obj('../obj_files/mountains.obj')
    position = np.array([0,0,0,0])
    object = Object(points,position,indices,cam,screen,normals,all_indices,maxval)
    position = np.array([-260,0,1000,0])
    #object2 = Object(points,position,indices,cam,screen,maxval)
    angle=math.pi/180
    held: bool = False
    #object.draw()
    rate = 10*60
    

    if testing:
        for _ in range(100):
            cam.rotate_cam(np.array([1,0,0]),angle)
            cam.move_cam([0,0,10,0])
            screen.fill("slate gray")
            object.draw4()
            pg.display.flip()
    if not testing:
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT: running = False
                elif event.type == pg.MOUSEBUTTONDOWN: held = True
                elif event.type == pg.MOUSEBUTTONUP: held = False

                if event.type == pg.MOUSEMOTION and held:
                    dx, dy = event.rel
                    # Rotate about y axis when mouse moves L->R and x-axis when mouse moves UP->DOWN
                    cam.rotate_cam(np.array([1,0,0]),angle*dy/10)
                    cam.rotate_cam(np.array([0,1,0]),-angle*dx/10)

            # CONTROLS
            keys = pg.key.get_pressed()
            # if want simultaneous movement change these all to ifs
            fps = clock.get_fps()
            if fps == 0:
                fps = 30
            if keys[pg.K_w]:
                cam.move_cam([0,0,rate/fps,0])
            elif keys[pg.K_s]:
                cam.move_cam([0,0,-rate/fps,0])
            if keys[pg.K_d]:
                cam.move_cam([rate/fps,0,0,0])
            elif keys[pg.K_a]:
                cam.move_cam([-rate/fps,0,0,0])
            if keys[pg.K_SPACE]:
                cam.move_cam([0,-rate/fps,0,0])
            elif keys[pg.K_LSHIFT]:
                cam.move_cam([0,rate/fps,0,0]) 
            elif keys[pg.K_t]:
                cam.third_person = not cam.third_person
                print("POV", cam.third_person)
            if keys[pg.K_UP]:
                cam.fovx += math.pi/180
                cam.update()
            elif keys[pg.K_DOWN]:
                cam.fovx -= math.pi/180
                cam.update()

            # DRAWING SCREEN
            # TODO maybe add a more efficient method to just draw over previous lines
            screen.fill("slate gray")
            object.draw4()
            #object2.draw()

            pg.display.flip()
            clock.tick(60)
            print(clock)


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main(True)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(40)

    #main()


pg.quit()