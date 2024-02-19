import math
import cProfile,pstats


from quaternions import *
from functions import *
from constants import *
from camera import *
from Object import *

def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pg.time.Clock() 
    running = True
    cam = Camera()
    position = np.array([0,0,0,0])
    object = Object(cam, screen, 'side_prop_housings.obj', position)
    angle=math.pi/180
    held = False

    
    

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


if __name__ == '__main__':
    main()
pg.quit()

