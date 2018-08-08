import numpy as np
import matplotlib.pyplot as plt
from sequential.plot import Visualizer
from sequential.utils import pickle_load, check_dir


if __name__ == '__main__':
    # loss function
    check_dir("./pick")
    loss = np.load("./result/draw_losses.npy")
    np.savetxt("./result/loss.csv", loss,delimiter=",")
    vis = Visualizer()
    sp = 0
    vis.dyplot(loss[sp:, 0], loss[sp:, 1], ["loss_separate", "Ld", "Lg"], "./result")

    vis.create_animated_gif("./pick/Train_***.png", "Test.gif", "./result")