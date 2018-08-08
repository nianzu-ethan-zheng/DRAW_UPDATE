import numpy as np
import tensorflow as tf


class ZoomableAttetionWindow(object):
    def __init__(self, channels, img_height, img_width, N):
        """A zoomable attention window for images

        channels, img_height, img_width : int
        N : attention window size
        """
        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width
        self.N = N

    def filterbank_matrices(self, center_y, center_x, delta, sigma2):
        """Create a Fy and Fx
        center_x, center_y : vector(shape: batch_size, 1) center position
        delta : vector(shape: batch_size, 1) position gap
        sigma : vector(shape: batch_size,1) position cover

        Return:
        Fy,Fx : shape: (batch_size, attention window size, img_height/width)
        """
        tol = 1e-9
        N = self.N

        rng = tf.reshape(tf.range(N, dtype=tf.float32) - N/2 + 0.5, [1, -1])  # 1xN

        muX = tf.reshape(center_x + rng * delta, [-1, N, 1])   # shape: batch_size x N x 1
        muY = tf.reshape(center_y + rng * delta, [-1, N, 1])

        a = tf.reshape(tf.range(self.img_width, dtype=tf.float32), [1, 1, -1])  # shape: 1x1xA
        b = tf.reshape(tf.range(self.img_height, dtype=tf.float32), [1, 1, -1])

        sigma2 = tf.reshape(sigma2, [-1, 1, 1])  # batch_size x 1 x 1

        FX = tf.exp(-tf.square(a-muX)/(2.*sigma2 + tol))  # batch_size x N x A
        FY = tf.exp(-tf.square(b-muY)/(2.*sigma2 + tol))
        FX = FX / tf.maximum(tf.reduce_sum(FX, axis=2, keepdims=True), tol)
        FY = FY / tf.maximum(tf.reduce_sum(FY, axis=2, keepdims=True), tol)

        # Get the position of region
        # Position oder: X: (height,width) (y, x), (B, A)
        # but in coordinates: width direction is the x_direction -> (Cx, Cy)
        RG = tf.ceil(tf.concat(
            [tf.maximum(center_x+(- N/2+0.5)*delta, 0),
             tf.maximum(center_y+(- N/2+0.5)*delta, 0),
             (N-1)*delta,
             (N-1)*delta], axis=1))

        return FY, FX, tf.to_int32(RG)

    def read(self, images, center_y, center_x, delta, sigma2):
        """Extract a batch of attention from given images
        Parameters
        ----------
        images: batch of images with shape ( batch size, img_size). Internally
                it will be reshaped to a (batch_size, img_height, img_width)
        center_x, center_y: Center coordinates for the attention window
        delta: distance between extracted grid points
        sigma: std. dev for Gaussian readout kernel; shape(batch_size, )
        """
        N = self.N
        channels = self.channels

        # Reshape input into 2d images
        I = tf.reshape(images, [-1, self.img_height, self.img_width])

        # Get filterbank
        FY, FX, Region= self.filterbank_matrices(center_y, center_x, delta, sigma2)

        FY = tf.tile(FY, [channels, 1, 1])
        FX = tf.tile(FX, [channels, 1, 1])

        W = tf.matmul(tf.matmul(FY, I), tf.transpose(FX, [0, 2, 1]))
        return tf.reshape(W, [-1, channels*N*N]), Region

    def write(self, windows, center_y, center_x, delta, sigma2):
        """Write a batch of window into full sized images
        Parameters
        ----------
        windows: batch of images with shape ( batch size*channels*N*N). Internally
                it will be reshaped to a (batch_size*channels, N, N)
        center_x, center_y: Center coordinates for the attention window
        delta: distance between extracted grid points
        sigma: std. dev for Gaussian readout kernel; shape(batch_size, )
        """
        N = self.N
        channels = self.channels

        # Reshape input into 2d windows
        W = tf.reshape(windows, [-1, N, N])

        FY, FX , Region= self.filterbank_matrices(center_y, center_x, delta, sigma2)

        FY = tf.tile(FY, [channels, 1, 1])
        FX = tf.tile(FX, [channels, 1, 1])

        # apply the filter
        I = tf.matmul(tf.matmul(tf.transpose(FY, perm=[0, 2, 1]), W), FX)
        return tf.reshape(I, (-1, channels*self.img_height*self.img_width)), Region

    def nn2att(self, l, is_test=False):
        """Convert neural-net outputs to attention parameters
        Parameters
        ------------------------
        l:  A batch of neural net outputs with shape(batch_size, 5)
        """
        center_y, center_x, log_delta , log_sigma2, log_gamma = tf.split(l, 5, 1)
        if is_test:
            return center_y, center_x, log_delta, log_sigma2, log_gamma

        # normalize coordinates
        center_y = (self.img_height + 1)/2 * (center_y + 1)
        center_x = (self.img_width + 1)/2 * (center_x + 1)
        delta = (max(self.img_height, self.img_width)-1) / (self.N-1) * tf.exp(log_delta)
        sigma2 = tf.exp(log_sigma2)
        gamma = tf.exp(log_gamma)

        return center_y, center_x, delta, sigma2, gamma

if __name__ == "__main__":
    from PIL import Image

    channels = 3
    height = 480
    width = 640

    I = Image.open("cat.jpg")
    print(np.array(I).shape)
    I = I.resize((640, 480))  # input tuple (width, height)| return shape: width x height x Channels
    print(np.array(I).shape)
    I = np.array(I).transpose([2, 0, 1])  # shape: C x H x W
    I = np.expand_dims(I, 0) / 255.
    print(I.shape)
    # -------------------------------------------------
    center_y = 250.5
    center_x = 330.5
    delta = 5.0
    sigma2 = 4.0
    gamma = 1
    N = 40

    out = np.array([center_y, center_x,delta, sigma2, gamma]).reshape(1, -1)
    print(out.shape)
    # #-----------------------------------------------
    images = tf.placeholder(tf.float32, [None, channels, height, width])
    l = tf.placeholder(tf.float32, [None, 5])
    att = ZoomableAttetionWindow(channels, height, width, N)
    center_y, center_x, delta, sigma2, gamma = att.nn2att(l, is_test=True)
    W, region_r = att.read(images, center_y, center_x, delta, sigma2)
    Img_, region_w = att.write(W, center_y, center_x, delta, sigma2)

    print(region_r.get_shape())

    # ------------------------------------------------
    sess = tf.InteractiveSession()
    (W, region_r) = sess.run((W, region_r), feed_dict={images:I, l:out})
    print(W.shape, region_r)
    I2, region_w = sess.run([Img_, region_w], feed_dict={images:I, l:out})
    print(region_w)

    def imagify(flat_image, h, w):
        image = flat_image.reshape([channels, h, w])
        image = image.transpose([1, 2, 0])
        return image/ image.max()

    import pylab
    import matplotlib.patches as patches
    fig, ax = pylab.subplots(1)
    ax.imshow(imagify(I, height, width), interpolation='nearest')
    rect = patches.Rectangle(region_r[0, 0:2], region_r[0, 2], region_r[0, 3], color='r', fill=False)
    ax.add_patch(rect)

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(W, N, N), interpolation='nearest')

    fig, ax = pylab.subplots(1)
    ax.imshow(imagify(I2, height, width), interpolation='nearest')
    rect = patches.Rectangle(region_w[0, 0:2], region_w[0, 2], region_w[0, 3], color='r', fill=False)
    ax.add_patch(rect)

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(I, height, width)-imagify(I2, height, width), interpolation='nearest')

    pylab.show(block=True)






































