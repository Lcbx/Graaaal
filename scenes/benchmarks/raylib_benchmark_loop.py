from pyray import *


update = lambda frameTime: None
draw = lambda: None
add_cubes = lambda: None
cubes_len = lambda: 0


rnd_uint8 = lambda : get_random_value(0, 255)
rnd_color = lambda : Color(rnd_uint8(),rnd_uint8(),rnd_uint8(),255)

SPACE_SIZE = 30
CUBE_MAX = 7


camera = Camera3D(
    Vector3(30, 70,-25),
    Vector3(0,0,-25),
    Vector3(0,1,0),
    60.0,
    CAMERA_PERSPECTIVE
)

def run():
    set_trace_log_level(LOG_WARNING)
    init_window(800, 450, "Hello")
    set_target_fps(60)

    frame_index = 0

    while not window_should_close():
        
        frameTime = get_frame_time()

        update(frameTime)
        
        begin_drawing()
        begin_mode_3d(camera)
        clear_background(WHITE)

        draw()
        
        end_mode_3d()
        draw_text(f"fps {get_fps()} cubes {cubes_len()} ", 10, 10, 20, LIGHTGRAY)
        end_drawing()
        
        frame_index+=1
        if frame_index%10 == 0 and get_fps() > 50: add_cubes()

    close_window()

