# Draw
A Recurrent Neural network for Image Generation, 2015

## DRAW Structure
![](https://github.com/DreamPurchaseZnz/DRAW_UPDATE/blob/master/Pic/DRAW_STRUCTURE.png)


## Attention 

![Draw](https://github.com/DreamPurchaseZnz/GAN_models/blob/master/Draw/Pic/Draw.png)

KL divergence and Cross Entropy are the drive force, putting the cycle more and more efficient

## Read & Write 
![RW](https://github.com/DreamPurchaseZnz/GAN_models/blob/master/Draw/Pic/Read%26Write.png)

Tips
- Filterbank matrices are determined by the precious decoder output


## Result
The canvas can not be start with zero, otherwise sigmoid(zero)=0.5 and the canvas will be gray at begaining.
Like the following 

![](Pic/z50-n40_Train.gif)
![](Pic/z50-n40_Test.gif)

However the generated images are misarable. Guess that the noise dimensions are blamed for the shapeless image 

Remark:
- learning rate decay(exponent) is vatal to stable the training process and its value can be very large such as 0.01

- when the size of model decrease, the learning rate need to decay very quickly, 
So the Epoch need to be decrease as well --> e.g. when the hidden size is 100, the max epoch is 50K.
By contrast, the epochs is 250K while while the size 10. 

- The reference process does not meet the expectation

![](Pic/z100-n64-wr25_Train.gif)

![](Pic/z100-n64-wr25_Test.gif)

Remark:
- the model is hard to converge
- whenever the training process perform good, it still have a bad result on test set



## Reference
Theano

- [jbornschein](https://github.com/jbornschein/draw) (**Recommend**)

Pytorch: 

- [chenzhaomin123](https://github.com/chenzhaomin123/draw_pytorch)
- [skaae](https://github.com/skaae/lasagne-draw)(**Recommend**)

Tensorflow: 
- [mnist](https://github.com/lovecambi/DRAW) (**Recommend**)
- [birds](https://github.com/hollygrimm/draw_birds)
- [1D](https://github.com/RobRomijnders/DRAW_1D)
- [Automatic GIF Generation](https://github.com/Singularity42/Sync-DRAW)
- [Automatic GIF Generation2](https://github.com/syncdraw/Sync-DRAW)
