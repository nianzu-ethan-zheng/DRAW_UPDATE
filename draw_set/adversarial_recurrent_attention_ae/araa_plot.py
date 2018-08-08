import numpy as np
import matplotlib.pyplot as plt
from sequential.plot import Visualizer
from sequential.utils import pickle_load, check_dir


if __name__ == '__main__':
    # loss function
    check_dir("./pick")
    loss = np.load("./result/draw_losses.npy")
    vis = Visualizer()
    vis.mtsplot(loss, name=["Loss Tendency", "Loss", "Lx", "Lz", "Cost", "Lg", "Ld"], dir="./result")
    sp = 0
    vis.tsplot(loss[sp:, 0], "Lx", "./result")
    vis.tsplot(loss[sp:, 1], "Lz", "./result")
    vis.dyplot(loss[sp:, 0], loss[sp:, 1], ["loss_separate", "Lx", "Lz"], "./result")
    vis.dyplot(loss[sp:, 3], loss[sp:, 4], ["loss_separate", "Lg", "Ld"], "./result")

    # Images
    [canvases_gen, rg_gen, canvases_trn, rg_trn], names = pickle_load("./result/draw.pkl")

    T, batch_size, img_size = canvases_gen.shape
    width = height = int(np.sqrt(img_size))
    # ------------------------------------------------------------------------------------------------
    check_dir("./pick", is_restart=True)
    # ------------------------------------------------------------------------------------------------
    choice = "pt"
    create_gif = True

    if choice == "qc":
        for t in range(T):
            vis.img_grid(canvases_gen[t, :, :], t, height, width, name=names[0])

        for t in range(T):
            vis.img_grid(canvases_trn[t, :, :], t, height, width, name=names[1])

    elif choice == "pt":
        for t, (x, r) in enumerate(zip(canvases_gen, rg_gen)):
            vis.annotated_img_grid(x, t, width, height, r, name=names[0])
            print("The ensemble images in {:d} iteration  have been saved".format(t))
        print("check these pictures in {}".format("pick file dictionary"))

        plt.clf()
        for t, (x, r) in enumerate(zip(canvases_trn, rg_trn)):
            vis.annotated_img_grid(x, t, width, height, r, name=names[1])
            print("The ensemble images in {:d} iteration  have been saved".format(t))
        print("check these pictures in {}".format("pick file dictionary"))
    else:
        pass

    if create_gif:
        vis.create_animated_gif("./pick/Train_**.png", "Train.gif", "./result")
        vis.create_animated_gif("./pick/Generate_**.png", "Test.gif", "./result")





