import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from core.hyperparameters import hp


class Plotter:
    def show_img(self, image, size=10, figname="example_image"):
        image = image / 2 + 0.5
        image = image.numpy()

        plt.figure(figsize=(size, size))
        plt.imshow(np.transpose(image, (1, 2, 0)))
        # plt.show()

        plt.savefig(f"{figname}.png")

    def to_img(self, x):
        return x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)

    def plot_output(self, path, x, y):
        img = mpimg.imread(path)

        plt.figure(figsize=(x, y))
        plt.imshow(img)
        plt.show()
