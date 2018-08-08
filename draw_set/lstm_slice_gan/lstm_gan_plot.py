import numpy as np
import matplotlib.pyplot as plt
from sequential.plot import Visualizer
from sequential.utils import pickle_load, check_dir


if __name__ == '__main__':
    # loss function
    check_dir("./pick")
    loss = np.load("./result/draw_losses.npy")
    vis = Visualizer()
    sp = 0
    vis.dyplot(loss[sp:, 0], loss[sp:, 1], ["loss_separate", "Ld", "Lg"], "./result")

    # Images
    [canvases_gen], names = pickle_load("./result/draw.pkl")
    batch_size, img_size = canvases_gen.shape
    width = height = int(np.sqrt(img_size))
    # ------------------------------------------------------------------------------------------------
    check_dir("./pick", is_restart=True)
    # ------------------------------------------------------------------------------------------------
    choice = "qc"
    create_gif = True

    vis.img_grid(canvases_gen, 0, height, width, name=names[0])
