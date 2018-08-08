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
from attnGAN.lstm_gan import *
from tensorflow.examples.tutorials import mnist


def main(name, dataset, epochs, batch_size, learning_rate, attention,
          n_iter, enc_dim, dec_dim, z_dim, dis_dim, load_from_file=False, is_test=False):

    if name is None:
        name = dataset

    channels, height, width = 1, 28, 28
    x_dim = channels*height*width

    # Configure attention mechanism
    if attention != "":
        read_N, write_N = attention.split(",")

        read_N = int(read_N)
        write_N = int(write_N)

        reader = AttentionReader_x(channels, height, width, read_N)
        writer = AttentionWriter(channels, height, width, write_N)
        attention_tag = "r%d-w%d" % (read_N, write_N)
    else:
        reader = Reader_x(channels, height, width)
        writer = Writer(x_dim, channels, height, width)

        attention_tag = "full"

    lr_str = "%1.0e" % learning_rate
    result_dir = os.path.join("./", "result")
    check_dir(result_dir, is_restart=False)
    check_dir("./process", is_restart=True)
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
    # --------------------------------------------------------------
    data_directory = os.path.join("../", "mnist")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train  # (samples, height, width)
    train_set = (train_data.images - 0.5)/0.5
    # c_init = np.tile(np.mean(train_data.images, axis=0), [batch_size, 1])
    # c_init = tf.convert_to_tensor(c_init)

    # ----------------------------------------------------------------------
    encoder_rnn = tf.nn.rnn_cell.LSTMCell(enc_dim, state_is_tuple=True)
    # lstm_dec = [tf.nn.rnn_cell.LSTMCell(dim, state_is_tuple=True) for dim in [dec_dim, dec_dim]]
    # decoder_rnn = tf.nn.rnn_cell.MultiRNNCell(lstm_dec,state_is_tuple=True)
    # lstm_dis = [tf.nn.rnn_cell.LSTMCell(dim, state_is_tuple=True) for dim in [dis_dim, dis_dim]]
    # dis_rnn = tf.nn.rnn_cell.MultiRNNCell(lstm_dis, state_is_tuple=True)
    decoder_rnn = tf.nn.rnn_cell.LSTMCell(dec_dim, state_is_tuple=True)
    dis_rnn = tf.nn.rnn_cell.LSTMCell(dis_dim, state_is_tuple=True)
    sampler = Qsampler(z_dim)

    lm = LstmModel(
        batch_size, x_dim, z_dim, dec_dim,
        n_iter=n_iter,
        reader=reader,
        encoder_rnn=encoder_rnn,
        sampler=sampler,
        decoder_rnn=decoder_rnn,
        writer=writer,
        discriminator_rnn=dis_rnn,
        dis_size=dis_dim)

    # -------------------------------------------------------------------------

    x = tf.placeholder(tf.float32, shape=(batch_size, x_dim))
    lr = tf.placeholder(tf.float32, shape=[])
    warm_up = tf.placeholder(tf.float32, shape=[])

    x_sample, g_vars, xs, r_ws = lm.generator(c_int=None)
    d_real, d_vars = lm.discriminator(x, ru=False)
    d_fake, _ = lm.discriminator(x_sample, ru=True)

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
            n_batch = 10000//batch_size + 1
            x_sams = []
            for _ in range(n_batch):
                canvases_gen = sess.run(x_sample, feed_dict={})
                x_sams.append(canvases_gen)

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


                ld, _ = sess.run(
                    [d_loss, d_solver], feed_dict={x: x_batch, lr: lr_epoch, warm_up: wu})

                # update G
                lg, _ = sess.run(
                    [g_loss, g_solver], feed_dict={x: x_batch, lr: lr_epoch, warm_up: wu})

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
                x_gen = sess.run(x_sample, feed_dict={})
                from sequential.plot import Visualizer
                vis = Visualizer()
                vis.img_grid(x_gen, epoch, height, width, path="./process", name="train")


        # -----------------------------------------------------------

        # canvases: shape(n_iter, batch_size, x_dim)
        # rg: shape(n_iter, batch_size, 4)
        canvases_gen, rg_gen = sess.run([xs, r_ws], feed_dict={})
        canvases_gen, rg_gen = np.array(canvases_gen), np.array(rg_gen)
        pickle_save([canvases_gen, rg_gen], ["Generate", "Train"], "./result/draw.pkl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                        default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=225, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=3E-4, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str,
                        default="5, 5",
                        help="Use attention mechanism (read_window,write_window)")
    parser.add_argument("--niter", type=int, dest="n_iter",
                        default=12, help="No. of iterations")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                        default=256, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                        default=256, help="Decoder  RNN state dimension")
    parser.add_argument("--dis-dim", type=int, dest="dis_dim",
                        default=256, help="Discriminator  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                        default=100, help="Z-vector dimension")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=200, help="Number of training epochs to do")
    args = parser.parse_args()

    main(load_from_file=False, is_test=False, **vars(args))


