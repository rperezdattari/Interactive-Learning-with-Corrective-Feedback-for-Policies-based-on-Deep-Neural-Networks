from pyglet.window import key
from tools.functions import str_2_array


class Feedback:
    def __init__(self, env, key_type='1', h_up='1,0', h_down='-1,0',
                 h_right='0,1', h_left='0,-1', h_null='0,0'):
        if key_type == '1':
            env.unwrapped.viewer.window.on_key_press = self.key_press
            env.unwrapped.viewer.window.on_key_release = self.key_release
        elif key_type == '2':
            env.unwrapped.window.on_key_press = self.key_press
            env.unwrapped.window.on_key_release = self.key_release
        else:
            print('No valid feedback type selected!')
            exit()

        self.h = str_2_array(h_null)  # Human correction
        self.h_null = str_2_array(h_null)
        self.h_up = str_2_array(h_up)
        self.h_down = str_2_array(h_down)
        self.h_right = str_2_array(h_right)
        self.h_left = str_2_array(h_left)
        self.restart = False

    def key_press(self, k, mod):
        if k == key.SPACE:
            self.restart = True
        if k == key.LEFT:
            self.h = self.h_left
        if k == key.RIGHT:
            self.h = self.h_right
        if k == key.UP:
            self.h = self.h_up
        if k == key.DOWN:
            self.h = self.h_down

    def key_release(self, k, mod):
        if k == key.LEFT or k == key.RIGHT or k == key.UP or k == key.DOWN:
            self.h = self.h_null

    def get_h(self):
        return self.h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done
