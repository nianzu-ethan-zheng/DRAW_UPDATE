"""
Author: Nianzu_Ethan_Zheng
Time: 2018-7-2
Address: Shenyang
"""
import numpy as np
import os
from argparse import ArgumentParser
from sequential.utils import pickle_save, check_dir
from sequential.process import Process
from sequential.learning_rate import create_lr_schedule, create_lr_pieces
from gan.gan_model import *
from tensorflow.examples.tutorials import mnist
from sequential.plot import Visualizer


def main(name, dataset, epochs, batch_size, learning_rate,
          dec_dim, z_dim, dis_dim, load_from_file=False, is_test=False):

    if name is None:
        name = dataset

    channels, height, width = 1, 28, 28
    x_dim = channels*height*width
    vis = Visualizer()
    # Configure attention mechanism

    lr_str = "%1.0e" % learning_rate
    result_dir = os.path.join("./", "result")
    check_dir(result_dir, is_restart=False)

    print("               dataset: %s" % dataset)
    print("          subdirectory: %s" % result_dir)
    print("   load from old model: %s" % load_from_file)
    print("            test_state: %s" % is_test)
    print("         learning rate: %g" % learning_rate)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    # ----------------------------------------------------------------------
    generator = Generator(dec_dim)
    discriminator = Discriminator(dis_dim)
    # -------------------------------------------------------------------------

    x = tf.placeholder(tf.float32, shape=(batch_size, x_dim))
    z = tf.placeholder(tf.float32, shape=[batch_size, z_dim])

    G_sample = generator(z)
    d_real = discriminator(x, reuse=False)
    d_fake = discriminator(G_sample, reuse=True)

    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
    # -----------------------------------------------------------------------
    optim_d = tf.train.AdamOptimizer(learning_rate)
    gradients_and_vars = optim_d.compute_gradients(d_loss, var_list=discriminator.vars)
    gradients_and_vars = [(grad_var[0] if grad_var[0] is None else tf.clip_by_norm(grad_var[0], 5),
                           grad_var[1]) for grad_var in gradients_and_vars]

    d_solver = optim_d.apply_gradients(gradients_and_vars)

    optim_g = tf.train.AdamOptimizer(learning_rate)
    gradients_and_vars = optim_g.compute_gradients(g_loss, var_list=generator.vars)
    gradients_and_vars = [(grad_var[0] if grad_var[0] is None else tf.clip_by_norm(grad_var[0], 5),
                           grad_var[1]) for grad_var in gradients_and_vars]

    g_solver = optim_g.apply_gradients(gradients_and_vars)

    # --------------------------------------------------------------
    data_directory = os.path.join("../", "mnist")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train  # (samples, height, width)
    train_set = (train_data.images - 0.5) / 0.5  # normalization; range: -1 ~ 1


    def sample_z(m,n):
        return np.random.normal(loc=0, scale=1, size=[m, n])
        # return np.random.uniform(-1,1, size=[m, n])

    # ----------------------------------------------------
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    saver = tf.train.Saver(
        max_to_keep=1
    )
    process = Process()

    with tf.Session(config=sess_config) as sess:
        '''
        Load from checkpoint or start a new session

        '''
        train_epoch_loss = []
        epoch_start = 0
        if load_from_file:
            # checkpoint without suffix
            saver.restore(sess, tf.train.latest_checkpoint(result_dir))
            train_epoch_loss = np.load("./result/draw_losses.npy")[:].tolist()
            epoch_start = len(train_epoch_loss)
        else:
            sess.run(tf.global_variables_initializer())

        if is_test:
            n_batch = 10000 // batch_size + 1
            x_sams = []
            for _ in range(n_batch):
                x_gen = sess.run(G_sample, feed_dict={z: sample_z(batch_size, z_dim)})
                x_sams.append(x_gen)

            x_sams = np.array(x_sams)
            print("The test result is saving")
            return pickle_save(x_sams, "Generate", "./result/draw_sample.pkl")

        for epoch in range(epoch_start, epochs):

            process.start_epoch()
            training_iteration_loss = []

            # Given data batch to Graph: 550 groups per epoch
            for batch_i in range(train_data.num_examples // batch_size):
                x_batch = train_set[batch_i * batch_size:(batch_i + 1) * batch_size]
                # x_batch, _ = train_data.next_batch(batch_size)
                # x_batch = np.float32(x_batch > 0.5)
                z_batch = sample_z(batch_size, z_dim)
                ld, _ = sess.run(
                    [d_loss, d_solver], feed_dict={x: x_batch, z:z_batch})

                # update G
                lg, _ = sess.run(
                    [g_loss, g_solver], feed_dict={z: sample_z(batch_size, z_dim)})

                training_iteration_loss.append([ld, lg])

            average_loss = np.mean(np.array(training_iteration_loss), axis=0)
            train_epoch_loss.append(average_loss)
            if epoch % 1 == 0 or epoch == epochs-1:
                process.format_meter(epoch, epochs, {
                    "Ld": average_loss[0],
                    "Lg": average_loss[1],
                })
            if (epoch % 10 == 0 and epoch != 0) or epoch == epochs-1:
                saver.save(sess, os.path.join(result_dir, "model_ckpt"), global_step=epoch)
                np.save(result_dir + "/draw_losses.npy", np.array(train_epoch_loss))
                print("Results have been saved")
                x_gen = sess.run(G_sample, feed_dict={z: sample_z(batch_size, z_dim)})
                vis.img_grid(x_gen, epoch, height, width, name="train")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=225, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-4, help="Learning rate")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                        default=2400, help="Decoder  RNN state dimension")
    parser.add_argument("--dis-dim", type=int, dest="dis_dim",
                        default=128, help="Discriminator  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=100, help="Z-vector dimension")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=800, help="Number of training epochs to do")
    args = parser.parse_args()

    main(load_from_file=False, is_test=False, **vars(args))


