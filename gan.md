# GAN
The picture come from [znxlwm](https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN)

![GAN](./Pic/tensorflow_GAN.png)

![DCGAN](./Pic/tensorflow_GAN.png)

Draw on the idea above, we find that: 

> The final layer of generator which is imposed on the sigmoid activation is hard to train. However if we change the activation to tanh,
the model is easy to train.

## Structure

Generator: 1200Relu-1200Relu-784Tanh

Discriminator: 240Maxout(5)-240Maxout(5)-1Sigmoid

## Result
![p](./Pic/Test.gif)

![p](./Pic/loss_gan.png)

![p](./Pic/train_440.png)


