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

    np.savetxt("./result/loss.csv", loss, delimiter=",")
    # Images
    [canvases_gen, rg_gen], names = pickle_load("./result/draw.pkl")

    T, batch_size, img_size = canvases_gen.shape
    width = height = int(np.sqrt(img_size))
    # ------------------------------------------------------------------------------------------------
    check_dir("./pick", is_restart=True)
    # ------------------------------------------------------------------------------------------------
    choice = "qc"
    create_gif = True

    if choice == "qc":
        for t in range(T):
            vis.img_grid(canvases_gen[t, :, :], t, height, width, name=names[0])

    elif choice == "pt":
        for t, (x, r) in enumerate(zip(canvases_gen, rg_gen)):
            vis.annotated_img_grid(x, t, width, height, r, name=names[0])
            print("The ensemble images in {:d} iteration  have been saved".format(t))
        print("check these pictures in {}".format("pick file dictionary"))

    else:
        pass

    if create_gif:
        vis.create_animated_gif("./pick/Generate_**.png", "Test.gif", "./result")