import numpy as np
import matplotlib.pyplot as plt


class FastImagePlot:
    def __init__(self, fig_num, observation, image_size, title_name, vmin=0, vmax=1):
        self.window = plt.figure(fig_num)
        self.image_size = image_size
        self.im = plt.imshow(np.reshape(observation, [self.image_size, self.image_size]),
                             cmap='gray', vmin=vmin, vmax=vmax)
        plt.show(block=False)
        self.window.canvas.set_window_title(title_name)
        self.window.canvas.draw()

    def refresh(self, observation):
        self.im.set_data(np.reshape(observation, [self.image_size, self.image_size]))
        self.window.draw_artist(self.im)
        self.window.canvas.blit()
        self.window.canvas.flush_events()
