"""
Author: Nianzu_Ethan_Zheng
Time: 2018-7-2
Address: Shenyang
"""
import numpy as np
import os
from argparse import ArgumentParser
from sequential.utils import pickle_save, check_dir, sliceinput, joinslice
from sequential.process import Process
from sequential.learning_rate import create_lr_schedule, create_lr_pieces
from lstm_slice_gan.s_lstm_gan import *
from tensorflow.examples.tutorials import mnist


def main(name, dataset, num_slice,epochs, batch_size, learning_rate,
         dec_dims, z_dim, dis_dims, load_from_file=False, is_test=False):

    if name is None:
        name = dataset

    channels, height, width = 1, 28, 28
    x_dim = channels*height*width

    patch_height, patch_width = height//num_slice, width//num_slice
    patch_dim = patch_height * patch_width

    n_iter = num_slice ** 2

    result_dir = os.path.join("./", "result")
    check_dir(result_dir, is_restart=False)

    print("               dataset: %s" % dataset)
    print("          subdirectory: %s" % result_dir)
    print("   load from old model: %s" % load_from_file)
    print("            test_state: %s" % is_test)
    print("         learning rate: %g" % learning_rate)
    print("          n_iterations: %d" % n_iter)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %s" % str(dec_dims))
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    # ----------------------------------------------------------------------
    decoder_rnn = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(dim, state_is_tuple=True) for dim in dec_dims])
    dis_rnn = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(dim, state_is_tuple=True) for dim in dis_dims])

    sampler = Qsampler(z_dim)

    lm = LstmModel(
        batch_size, x_dim, z_dim, dec_dims,
        n_iter=n_iter,
        sampler=sampler,
        decoder_rnn=decoder_rnn,
        discriminator_rnn=dis_rnn,
        dis_size=dis_dims, patch_size=patch_dim)

    # -------------------------------------------------------------------------

    x = tf.placeholder(tf.float32, shape=(n_iter, batch_size,patch_dim))
    lr = tf.placeholder(tf.float32, shape=[])
    warm_up = tf.placeholder(tf.float32, shape=[])

    x_samples, g_vars = lm.generator()
    d_real, d_vars = lm.discriminator(x, reuse=False)
    d_fake, _ = lm.discriminator(x_samples, reuse=True)

    d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
    # -----------------------------------------------------------------------
    optim_d = tf.train.AdamOptimizer(lr, beta1=0.75)
    gradients_and_vars = optim_d.compute_gradients(d_loss, var_list=d_vars)
    gradients_and_vars = [(grad_var[0] if grad_var[0] is None else tf.clip_by_norm(grad_var[0], 5),
                           grad_var[1]) for grad_var in gradients_and_vars]

    d_solver = optim_d.apply_gradients(gradients_and_vars)

    optim_g = tf.train.AdamOptimizer(lr, beta1=0.75)
    gradients_and_vars = optim_g.compute_gradients(g_loss, var_list=g_vars)
    gradients_and_vars = [(grad_var[0] if grad_var[0] is None else tf.clip_by_norm(grad_var[0], 5),
                           grad_var[1]) for grad_var in gradients_and_vars]

    g_solver = optim_g.apply_gradients(gradients_and_vars)

    # --------------------------------------------------------------
    data_directory = os.path.join("../", "mnist")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train  # (samples, height, width)
    train_set = (train_data.images - 0.5) / 0.5  # normalization; range: -1 ~ 1

    # ----------------------------------------------------
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    saver = tf.train.Saver(
        max_to_keep=10
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
            n_batch = 10000//batch_size + 1
            x_sams = []
            for _ in range(n_batch):
                canvases_gen = sess.run(x_samples, feed_dict={})
                transf = (patch_height, patch_width, height // patch_height, width // patch_width)
                cans = joinslice(canvases_gen, transf)
                x_sams.append(cans.reshape(batch_size, -1))

            x_sams = np.array(x_sams)
            print("The test result is saving")
            return pickle_save(x_sams, "Generate", "./result/draw_sample.pkl")
        
        lr_schedule = create_lr_pieces([2000, 10000, 30000], ["c", "e", "c"], base=learning_rate, rate=0.1)
        wu_schedule = create_lr_schedule(lr_base=1, decay_rate=2, decay_epochs=20000,
                                         truncated_epoch=20000, start_epoch=None, mode="constant")

        for epoch in range(epoch_start, epochs):

            process.start_epoch()
            lr_epoch = lr_schedule.apply(epoch)
            wu = wu_schedule(epoch)
            training_iteration_loss = []

            # Given data batch to Graph: 550 groups per epoch
            for batch_i in range(train_data.num_examples // batch_size):
                x_batch = train_set[batch_i * batch_size:(batch_i + 1) * batch_size]
                # x_batch, _ = train_data.next_batch(batch_size)
                # x_batch = np.float32(x_batch > 0.5)
                x_batch = np.reshape(x_batch, [batch_size, height, width])
                x_batch, tranf = sliceinput(x_batch, patch_height, patch_width)

                ld, _ = sess.run(
                    [d_loss, d_solver], feed_dict={x: x_batch, lr: lr_epoch, warm_up: wu})

                # update G
                k = 1
                lgs = []
                for _ in range(k):
                    lg, _ = sess.run(
                        [g_loss, g_solver], feed_dict={lr: lr_epoch, warm_up: wu})
                    lgs.append(lg)
                lg = np.mean(np.array(lgs))

                training_iteration_loss.append([ld, lg])

            average_loss = np.mean(np.array(training_iteration_loss), axis=0)
            train_epoch_loss.append(average_loss)
            if epoch % 1 == 0 or epoch == epochs-1:
                process.format_meter(epoch, epochs, {
                    "Ld": average_loss[0],
                    "Lg": average_loss[1],
                })
            if (epoch % 2 == 0 and epoch != 0) or epoch == epochs-1:
                saver.save(sess, os.path.join(result_dir, "model_ckpt"), global_step=epoch)
                np.save(result_dir + "/draw_losses.npy", np.array(train_epoch_loss))

                print("Results have been saved")

        # -----------------------------------------------------------

        # canvases: shape(n_iter, batch_size, x_dim)
        # rg: shape(n_iter, batch_size, 4)
        canvases_gen= sess.run(x_samples, feed_dict={})
        transf = (patch_height, patch_width, height//patch_height, width//patch_width)
        canvases_gen= joinslice(np.array(canvases_gen), transf)
        pickle_save([np.reshape(canvases_gen, [batch_size, -1])], ["Generate", "Train"], "./result/draw.pkl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=225, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1E-4, help="Learning rate")
    parser.add_argument("--num_slice", type=int, dest="num_slice",
                        default=2, help="No. of iterations")
    parser.add_argument("--dec-dim", type=int, dest="dec_dims",
                        default=[1000]*2, help="Decoder  RNN state dimension")
    parser.add_argument("--dis-dim", type=int, dest="dis_dims",
                        default=[1000]*2, help="Discriminator  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=100, help="Z-vector dimension")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=10, help="Number of training epochs to do")
    args = parser.parse_args()

    main(load_from_file=False, is_test=False, **vars(args))


