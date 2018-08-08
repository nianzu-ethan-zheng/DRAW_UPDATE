import numpy as np
import matplotlib.pyplot as plt
from sequential.plot import Visualizer
from sequential.utils import pickle_load, check_dir

vis = Visualizer()
check_dir("./pick", is_restart=True)
# ----------------------------------------------------
[x_sample, x_recons], _ = pickle_load("./result/ae.pkl")
batch_size, img_size = x_sample.shape
width = height = int(np.sqrt(img_size))

# Generate process
img = vis.img_grid(x_sample, 1, height, width, 'generate')

# Training process
vis.img_grid(x_recons, 1, height, width, "train")


# ----------------------------------------------------------
train_loss, name = pickle_load("./result/loss.pkl")
vis.mtsplot(np.array(train_loss), name=["Loss", "Cost", "Lx", "Lz"], dir="./result")
