__version__ = 0.1
__appname__ = "UVLP Software"

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

import numpy as np
import cv2
import screeninfo




if __name__ == "__main__":
    # print("hello")
    # dpg.create_context()
    # dpg.create_viewport(title=f"{__appname__} {__version__}", width=600, height=600)

    # # demo.show_demo()
    
    # with dpg.window(label="DMD Image", width=800, height=800, pos=(100, 100), tag="_image_out"):
    #     dpg.add_button(label="Full screen", callback=lambda:dpg.toggle_viewport_fullscreen())
    
    # dpg.create_viewport(title=f"{__appname__}3 {__version__}", width=600, height=600)


    # dpg.setup_dearpygui()
    # dpg.show_viewport()
    # dpg.start_dearpygui()
    # dpg.destroy_context()
    
    screen_id = 0
    is_image = True
    is_color = True

    # get the size of the screen
    print(screeninfo.get_monitors())
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    if not is_image:
        # create image
        if is_color:
            image = np.ones((height, width, 3), dtype=np.float32)
            image[:50, :50] = 0  # black at top-left corner
            image[height - 50:, :50] = [1, 0, 0]  # blue at bottom-left
            image[:50, width - 50:] = [0, 1, 0]  # green at top-right
            image[height - 50:, width - 50:] = [0, 0, 1]  # red at bottom-right
        else:
            image = np.ones((height, width), dtype=np.float32)
            image[0, 0] = 0  # top-left corner
            image[height - 2, 0] = 0  # bottom-left
            image[0, width - 2] = 0  # top-right
            image[height - 2, width - 2] = 0  # bottom-right
    else:
        # image = cv2.imread('wallpaper.jpg')
        image = cv2.imread('pattern.png')
        # image = cv2.imread('check.png')
        # image = cv2.imread('calib.jpg')
        # print(type(image), image)
        
        
    window_name = 'dmd_screen'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()