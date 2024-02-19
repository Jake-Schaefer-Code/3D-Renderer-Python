import math
import cProfile,pstats
import memory_profiler

from quaternions import *
from functions import *
from constants import *
from camera import *
from Object import *

def main(testing=True):
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pg.time.Clock() 
    running = True
    cam = Camera()
    position = np.array([0,0,0,0])
    object = Object(cam, screen, 'side_prop_housings.obj', position)
    angle=math.pi/180
    held = False
    #compile_functions()

    if testing:
        profiler = cProfile.Profile()
        profiler.enable()
        """cam = Camera()
        position = np.array([0,0,0,0])
        object = Object(cam, screen, 'side_prop_housings.obj', position)
        angle=math.pi/180
        held = False"""

        for i in range(100):
            cam.rotate_cam(np.array([1,0,0]),angle)
            cam.move_cam([0,0,0.025,0])
            screen.fill("slate gray")

            object.draw()
            pg.display.flip()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(40)
        #print('NEWTIME', object.newtime)
        #print('OLDTIME', object.oldtime)
    
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
                    cam.rotate_cam(np.array([0,1,0]),angle*dx/10)
            cam.check_movement()
            
            # DRAWING SCREEN
            screen.fill(BACKGROUND_COLOR)
            object.draw()

            pg.display.flip()
            clock.tick(60)
            print(clock)


if __name__ == '__main__':
    main(True)
pg.quit()

