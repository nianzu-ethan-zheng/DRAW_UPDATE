import numpy as np
import matplotlib.pyplot as plt
from plot import Visualizer
from matplotlib import gridspec, patches
from utils import pickle_load


def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale


def img_grid(imgs,
             iter,
             width, height,
             region,
             name="draw"):
    """Plot the images with position where the picture changes
    --------------------------------------------
    iter              i th iteration
    x                 Shape:(batch_size, x_dims)
    width, height     Figure size
    ------
    """
    plt.close("all")
    batch_size = imgs.shape[0]
    N = int(np.sqrt(batch_size))

    fig = plt.figure(num=1, figsize=(8, 8))
    gs = gridspec.GridSpec(N, N, wspace=0.1, hspace=0.1)
    ax = []
    for n, img in enumerate(imgs):
        row = n // N
        col = n % N

        ax.append(fig.add_subplot(gs[row, col]))
        img = scale_norm(img).reshape(height, width)
        ax[-1].imshow(img, cmap=plt.cm.gray)
        rect = patches.Rectangle(
            region[n, 0:2], region[n, 2], region[n, 3], color='r', fill=False)
        ax[-1].add_patch(rect)
        ax[-1].set_axis_off()

    plt.savefig("./pick/%s_%s.png" % (name, str(iter).zfill(2)))


def xrecons_grid(X, B, A):
    """Create a canvas to picture all the example  very quickly
    """
    plt.close('all')
    padsize = 1
    padval = .5
    ph = B + 2 * padsize
    pw = A + 2 * padsize
    batch_size = X.shape[0]
    N = int(np.sqrt(batch_size))
    X = X.reshape((N, N, B, A))
    img = np.ones((N * ph, N * pw)) * padval
    for i in range(N):
        for j in range(N):
            startr = i * ph + padsize
            endr = startr + B
            startc = j * pw + padsize
            endc = startc + A
            img[startr:endr, startc:endc] = X[i, j, :, :]

    return img


if __name__ == '__main__':
    # loss function
    loss = np.load("./result/draw_losses.npy")
    vis = Visualizer()
    vis.mtsplot(loss, "Loss Tendency", "loss", "./result")
    vis.tsplot(loss[100:, 0], "Lx", "./result")
    vis.tsplot(loss[100:, 1], "Lz", "./result")
    vis.dyplot(loss[100:, 0], loss[:, 1], ["loss_separate", "Lx", "Lz"], "./result")

    # Images
    [canvases_gen, rg_gen, canvases_trn, rg_trn], names = pickle_load("./result/draw.pkl")

    T, batch_size, img_size = canvases_gen.shape
    width = height = int(np.sqrt(img_size))
    quick = True
    # if quick:
    #     for t in range(T):
    #         img = xrecons_grid(canvases_gen[t, :, :], height, width)
    #         plt.matshow(img, cmap=plt.cm.gray)
    #         imgname = './pick/%s_%d.png' % (
    #         'draw_generate', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
    #         plt.savefig(imgname)
    #         print(imgname)
    #
    #     for t in range(T):
    #         img = xrecons_grid(canvases_trn[t, :, :], height, width)
    #         plt.matshow(img, cmap=plt.cm.gray)
    #         imgname = './pick/%s_%d.png' % (
    #             'draw_train', t)  # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
    #         plt.savefig(imgname)
    #         print(imgname)
    # else:
    #     for t, (x, r) in enumerate(zip(canvases_gen, rg_gen)):
    #         img_grid(x, t, width, height, r, name=names[0])
    #         print("The ensemble images in {:d} iteration  have been saved" .format(t))
    #     print("check these pictures in {}".format("pick file dictionary"))
    #
    #     plt.clf()
    #     for t, (x, r) in enumerate(zip(canvases_trn, rg_trn)):
    #         img_grid(x, t, width, height, r, name=names[1])
    #         print("The ensemble images in {:d} iteration  have been saved" .format(t))
    #     print("check these pictures in {}".format("pick file dictionary"))
    #
    # create_gif = True
    # if create_gif:
    #     vis.create_animated_gif("./pick/Train_**.png", "Train.gif", "./result")
    #     vis.create_animated_gif("./pick/Generate_**.png", "Test.gif", "./result")


