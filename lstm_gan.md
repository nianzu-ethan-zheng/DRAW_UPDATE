# LSTM GAN

## tanh & c0 = [0]
```
n_iter = 10
```

![](/Pic/lstm_gan/loss_tanh0.png)

![](/Pic/lstm_gan/tanh0.gif)

![](/Pic/lstm_gan/Generate_009.png)

There are some problems：

- model collapse: for example there aren't 4, 5, etc. in the image
- the balance between the G and D is not proper. the optimal D loss should be 0.693
