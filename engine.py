import cProfile,pstats
from quaternions import *
from functions import *
from constants import *
from camera import *
from Object import *


class RenderEngine:
    def __init__(self, testing: bool = False) -> None:
        self.is_running = True
        self.is_test = testing
        self.cam = Camera()
        self.objects: list[Object] = []
        self.setup_screen()

    def add_object(self, object: Object):
        self.objects.append(object)

    def setup_screen(self):
        pg.init()
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pg.time.Clock() 


    def run(self):
        if self.is_test:
            self._run_test()
        else:
            self._run_normal()

    def _run_normal(self):
        self.held = False
        while self.is_running:
            for event in pg.event.get():
                if event.type == pg.QUIT: 
                    self.is_running = False
                elif event.type == pg.MOUSEBUTTONDOWN: 
                    self.held = True
                elif event.type == pg.MOUSEBUTTONUP: 
                    self.held = False

                if event.type == pg.MOUSEMOTION and self.held:
                    dx, dy = event.rel
                    # Rotate about y axis when mouse moves L->R and x-axis when mouse moves UP->DOWN
                    self.cam.rotate(np.array([1,0,0]),ROT_ANGLE*dy/10)
                    self.cam.rotate(np.array([0,1,0]),ROT_ANGLE*dx/10)

            self.cam.check_movement()
            self.screen.fill(BACKGROUND_COLOR)

            # DRAWING OBJECTS
            for object in self.objects:
                object.draw()

            pg.display.flip()
            self.clock.tick(60)
            

    def _run_test(self):
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(100):
            self.cam.rotate(np.array([1,0,0]),ROT_ANGLE)
            self.cam.rotate(np.array([0,1,0]),ROT_ANGLE)
            self.cam.move_cam([0,0,0.02,0])
            self.screen.fill("slate gray")
            for object in self.objects:
                object.draw()
            pg.display.flip()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(40)


    def update_screen(self):
        # DRAWING SCREEN
        self.screen.fill(BACKGROUND_COLOR)

        # DRAWING OBJECTS
        for object in self.objects:
            object.draw()

        pg.display.flip()
        self.clock.tick(60)