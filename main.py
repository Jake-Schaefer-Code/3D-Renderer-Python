from quaternions import *
from functions import *
from constants import *
from camera import *
from Object import *
from engine import *

def main(testing=True, old=False):
    if old:
        from old_files import oldmain
        oldmain.old_main(testing)
    else:
        engine = RenderEngine(testing)
        object = Object(engine.cam, engine.screen, 'side_prop_housings.obj', np.array([0,0,0,0]))
        engine.add_object(object)
        engine.setup_screen()
        engine.run()

if __name__ == '__main__':
    main(testing=False, old=False)


