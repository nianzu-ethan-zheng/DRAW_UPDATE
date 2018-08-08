import numpy as np
import os
import matplotlib.pyplot as plt
from sequential.plot import Visualizer
from sequential.utils import pickle_load, check_dir
from sequential.metric import *
from tensorflow.examples.tutorials import mnist
# ---------------------------------------------------------------------------------------------------------
x_sams, names = pickle_load("./result/draw_sample.pkl")
x_dim = x_sams.shape[-1]
x_sams = x_sams.reshape([-1, x_dim])
x_sams = x_sams[:10000]

data_directory = os.path.join("../", "mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

test_data = mnist.input_data.read_data_sets(data_directory).test
valid = test_data.images[:1000]

sigma = 0.16681005
batch_size = 100
print("Using Sigma: {}".format(sigma))
parzen = parzen_estimation(x_sams, sigma)
ll = get_ll(valid, parzen, batch_size=batch_size)
se = np.std(ll)/np.sqrt(test_data.num_examples)

print("Log_likelihood of test set = {},se,{}".format(np.mean(ll), se))