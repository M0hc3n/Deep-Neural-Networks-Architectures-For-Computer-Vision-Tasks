import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Plotter:
    def imshow(img, size=10):
        img = img / 2 + 0.5  # de-normalizing
        npimg = img.numpy()
        plt.figure(figsize=(size, size))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def visualise_output(path, x, y):
        img = mpimg.imread(path)
        plt.figure(figsize=(x, y))
        plt.imshow(img)
        plt.show()
