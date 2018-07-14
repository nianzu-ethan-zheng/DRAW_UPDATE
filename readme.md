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


![](Pic/z50-n40_Train.gif=250x)

  
<p align="center">
<img, src="Pic/z50-n40_Test.gif" width=200>
</p>

However the generated images are misarable. Guess that the noise dimensions are blamed for the shapeless image 

Remark:
- learning rate decay(exponent) is vatal to stable the training process and its value can be very large such as 0.01

- when the size of model decrease, the learning rate need to decay very quickly, 
So the Epoch need to be decrease as well --> e.g. when the hidden size is 100, the max epoch is 50K.
By contrast, the epochs is 250K while while the size 10. 

- The reference process does not meet the expectation

Finally the cause I found will be explained in the next section.

# Draw_update
![](Pic/Draw_update.png)

- Add labels to decoder
- Add a new network to recognize the labels

# Random variable 
Here I write some [explanition for ericjang's poor result](https://github.com/DreamPurchaseZnz/Tensorflow_Learning/blob/master/Constants,%20Sequences,%20and%20Random%20Values.md)
because of the random variable defination is a mistake.

![Variable](https://github.com/DreamPurchaseZnz/Tensorflow_Learning/blob/master/Pic/random.png)

## Learning rate
Bigger learning rate is not always better. Despite the fact that big learning rate can speed the learning process, it also
can cause the osillation between the optim. It is hard to converge.

![](Pic/adaptive_lr.png)


## result of draw

![](Pic/draw/Train.gif)

![](Pic/draw/Test.gif)

Remark:

- the figure in the test.gif has some unknown numbers.






## Reference
Theano

- [jbornschein](https://github.com/jbornschein/draw) (**Recommend**)

Pytorch: 

- [chenzhaomin123](https://github.com/chenzhaomin123/draw_pytorch)
- [skaae](https://github.com/skaae/lasagne-draw)(**Recommend**)

Tensorflow: 
- [ericjang](https://github.com/ericjang/draw)(A mistake about random variable defination)
- [mnist](https://github.com/lovecambi/DRAW) (**Recommend**)
- [birds](https://github.com/hollygrimm/draw_birds)
- [1D](https://github.com/RobRomijnders/DRAW_1D)
- [Automatic GIF Generation](https://github.com/Singularity42/Sync-DRAW)
- [Automatic GIF Generation2](https://github.com/syncdraw/Sync-DRAW)
