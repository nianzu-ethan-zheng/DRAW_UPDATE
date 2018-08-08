from attnGAN.lstm_gan import LstmModel
from sequential.model_component import AttentionReader_x, AttentionWriter, Qsampler
import tensorflow as tf

channels, height, width = 1, 28, 28
x_dim = channels*height*width
read_N = 5
write_N = 5
reader = AttentionReader_x(channels, height, width, read_N)
writer = AttentionWriter(channels, height, width, write_N)
encoder_rnn = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
decoder_rnn = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
dis_rnn = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)
sampler = Qsampler(2)

m = LstmModel(batch_size=100, img_size=784,
              z_size=2, dec_size=256, n_iter=10,
              reader=reader,encoder_rnn=encoder_rnn,
              sampler=sampler, decoder_rnn=decoder_rnn, writer=writer,
              discriminator_rnn=dis_rnn)

x_pre, vars = m.generator()
print(x_pre.get_shape)
[print(var) for var in vars]

x = tf.placeholder(dtype=tf.float32, shape=[100, 784])
d, vars_d = m.discriminator(x_pre, ru=False)

print(d.get_shape)
[print(var) for var in vars_d]