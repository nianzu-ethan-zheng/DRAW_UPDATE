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
from draw.draw_model import *
from tensorflow.examples.tutorials import mnist


def main(name, dataset, epochs, batch_size, learning_rate, attention,
          n_iter, enc_dim, dec_dim, z_dim, load_from_file=False, is_test=False):

    if name is None:
        name = dataset

    channels, height, width = 1, 28, 28
    x_dim = channels*height*width

    # Configure attention mechanism
    if attention != "":
        read_N, write_N = attention.split(",")

        read_N = int(read_N)
        write_N = int(write_N)

        reader = AttentionReader(channels, height, width, read_N)
        writer = AttentionWriter(channels, height, width, write_N)
        attention_tag = "r%d-w%d" % (read_N, write_N)
    else:
        reader = Reader(channels, height, width)
        writer = Writer(x_dim, channels, height, width)

        attention_tag = "full"

    lr_str = "%1.0e" % learning_rate
    result_dir = os.path.join("./", "result")
    check_dir(result_dir, is_restart=False)
    longname = "%s-%s-t%d-enc%d-dec%d-z%d-lr%s" % \
               (dataset, attention_tag, n_iter, enc_dim, dec_dim, z_dim, lr_str)

    print("\nRunning experiment %s" % longname)
    print("               dataset: %s" % dataset)
    print("          subdirectory: %s" % result_dir)
    print("   load from old model: %s" % load_from_file)
    print("            test_state: %s" % is_test)
    print("         learning rate: %g" % learning_rate)
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    # ----------------------------------------------------------------------

    encoder_rnn = tf.nn.rnn_cell.LSTMCell(enc_dim, state_is_tuple=True)
    decoder_rnn = tf.nn.rnn_cell.LSTMCell(dec_dim, state_is_tuple=True)
    sampler = Qsampler(z_dim)

    draw = DrawModel(
        batch_size, x_dim, z_dim, dec_dim,
        n_iter=n_iter,
        reader=reader,
        encoder_rnn=encoder_rnn,
        sampler=sampler,
        decoder_rnn=decoder_rnn,
        writer=writer)

    # -------------------------------------------------------------------------

    x = tf.placeholder(tf.float32, shape=(batch_size, x_dim))
    lr = tf.placeholder(tf.float32, shape=[])
    warm_up = tf.placeholder(tf.float32, shape=[])

    x_recons, kl_term, xs, c_ws, zs = draw.reconstruct(x)
    sample_x, sample_rg, sample_zs = draw.sample()

    def binary_crossentropy(t, o):
        eps = 1E-8
        return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))

    Lx = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x, tf.nn.sigmoid(x_recons)), 1))

    # Lx = tf.losses.sigmoid_cross_entropy(x, logits=x_recons)
    Lz = tf.reduce_mean(kl_term)
    cost = Lx + Lz * warm_up

    # -----------------------------------------------------------------------
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    gradients_and_vars = optim.compute_gradients(cost)
    gradients_and_vars = [(grad_var[0] if grad_var[0] is None else tf.clip_by_norm(grad_var[0], 10),
                           grad_var[1]) for grad_var in gradients_and_vars]

    train_opt = optim.apply_gradients(gradients_and_vars)

    # --------------------------------------------------------------
    data_directory = os.path.join("../", "mnist")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train  # (samples, height, width)

    # ----------------------------------------------------
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    saver = tf.train.Saver(
        max_to_keep=2
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
                canvases_gen = sess.run(sample_x, feed_dict={})
                x_sams.append(canvases_gen[-1])

            x_sams = np.array(x_sams)
            print("The test result is saving")
            return pickle_save(x_sams, "Generate", "./result/draw_sample.pkl")

        lr_schedule = create_lr_pieces([5, 8, 30000], ["c", "e", "c"], base=4E-3, rate=0.1)
        wu_schedule = create_lr_schedule(lr_base=1, decay_rate=2, decay_epochs=20000,
                                         truncated_epoch=20000, start_epoch=None, mode="constant")

        for epoch in range(epoch_start, epochs):

            process.start_epoch()
            lr_epoch = lr_schedule.apply(epoch)
            wu = wu_schedule(epoch)
            training_iteration_loss = []

            # Given data batch to Graph: 550 groups per epoch
            for batch_i in range(train_data.num_examples // batch_size):

                x_batch, _ = train_data.next_batch(batch_size)
                x_batch = np.float32(x_batch > 0.5)

                lx, lz, tl, _ = sess.run(
                    [Lx, Lz, cost, train_opt], feed_dict={x: x_batch, lr: lr_epoch, warm_up: wu})

                training_iteration_loss.append([lx, lz, tl])

            average_loss = np.mean(np.array(training_iteration_loss), axis=0)
            train_epoch_loss.append(average_loss)
            if epoch % 1 == 0 or epoch == epochs-1:
                process.format_meter(epoch, epochs, {
                    "Lx": average_loss[0],
                    "Lz": average_loss[1],
                    "Cost": average_loss[2]
                })
            if (epoch % 10 == 0 and epoch != 0) or epoch == epochs-1:
                saver.save(sess, os.path.join(result_dir, "model_ckpt"), global_step=epoch)
                np.save(result_dir + "/draw_losses.npy", np.array(train_epoch_loss))

                print("Results have been saved")

        # -----------------------------------------------------------

        # canvases: shape(n_iter, batch_size, x_dim)
        # rg: shape(n_iter, batch_size, 4)
        x_batch, _ = train_data.next_batch(batch_size)
        x_batch = np.float32(x_batch > 0.5)
        canvases_gen, rg_gen, zs_gen = sess.run([sample_x, sample_rg, sample_zs], feed_dict={})
        canvases_trn, rg_trn, zs_trn = sess.run([xs, c_ws, zs], feed_dict={x: x_batch})
        canvases_gen, rg_gen, zs_gen = np.array(canvases_gen), np.array(rg_gen), np.array(zs_gen)
        canvases_trn, rg_trn, zs_trn = np.array(canvases_trn), np.array(rg_trn), np.array(zs_trn)
        pickle_save([canvases_gen, rg_gen, canvases_trn, rg_trn], ["Generate", "Train"], "./result/draw.pkl")
        pickle_save([zs_gen, zs_trn], ["Generate, Train"], "./result/draw_latent.pkl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=4E-3, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str,
                        default="5,5",
                        help="Use attention mechanism (read_window,write_window)")
    parser.add_argument("--niter", type=int, dest="n_iter",
                        default=32, help="No. of iterations")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                        default=256, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                        default=256, help="Decoder  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=100, help="Z-vector dimension")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=10, help="Number of training epochs to do")
    args = parser.parse_args()

    main(load_from_file=True, is_test=True, **vars(args))


