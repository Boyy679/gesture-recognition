# gesture-recognition

This part mainly have 4 parts:

* CNN model for ASL dataset gesture recognition
* ASL augmentation to increase the validation acracy
* ASL prediction
* ASL PC camera gesture recognition



---------

### CNN mode for ASL dataset gesture recognition

* Data

Reshape our dataset so that they are in a 28x28 pixel format.   This will allow our convolutions to associate groups of pixels and detect important features.

* CNN model

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_97 (Conv2D)           (None, 28, 28, 100)       1000      
_________________________________________________________________
batch_normalization_65 (Batc (None, 28, 28, 100)       400       
_________________________________________________________________
conv2d_98 (Conv2D)           (None, 28, 28, 75)        67575     
_________________________________________________________________
dropout_51 (Dropout)         (None, 28, 28, 75)        0         
_________________________________________________________________
conv2d_99 (Conv2D)           (None, 28, 28, 50)        33800     
_________________________________________________________________
max_pooling2d_78 (MaxPooling (None, 10, 10, 50)        0         
_________________________________________________________________
batch_normalization_66 (Batc (None, 10, 10, 50)        200       
_________________________________________________________________
dropout_52 (Dropout)         (None, 10, 10, 50)        0         
_________________________________________________________________
max_pooling2d_79 (MaxPooling (None, 5, 5, 50)          0         
_________________________________________________________________
conv2d_100 (Conv2D)          (None, 5, 5, 50)          22550     
_________________________________________________________________
batch_normalization_67 (Batc (None, 5, 5, 50)          200       
_________________________________________________________________
...
Total params: 136,549.0
Trainable params: 136,149.0
Non-trainable params: 400.0
_________________________________________________________________
```

The training accuracy is very high, and the validation accuracy has improved as well. This is a great result, as all we had to do was swap in a new model. But the validation accuracy jumping around. This is an indication that our model is still not generalizing perfectly. 

```
Epoch 25/25
27455/27455 [==============================] - 20s - loss: 0.0027 - acc: 0.9991 - val_loss: 0.0855 - val_acc: 0.9844
```

We will use augmentation to increase the model generalization in the following.

### ASL augmentation

After we train CNN model for gesture, we will find the validation accuracy is still lower than train accuracy. So the following is introduce the [**data augmentation**](https://link.springer.com/article/10.1186/s40537-019-0197-0)(a useful technique for many deep learning applications), which is to teach our model to be more robust when looking at new data, programmatically increase the size and variance in our dataset.

The increase in size gives the model more images to learn from while training.  The increase in variance helps the model ignore unimportant features and select only the features that are truly important in classification, allowing it to generalize better.

Before compiling the model, it's time to set up data augmentation.

Keras comes with an image augmentation class called `ImageDataGenerator`. We recommend checking out the [documentation here](https://keras.io/api/preprocessing/image/#imagedatagenerator-class). It accepts a series of options for augmenting your data. 

* Batch size
  * Another benefit of the `ImageDataGenerator` is that it [batches](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/) our data so that our model can train on a random sample.
  * If the model is [truly random](http://sites.utexas.edu/sos/random/), meaning that the data is properly shuffled so it's fair like a deck of cards, then our sample can do a good job of representing all of our data even though it is a tiny fraction of the population. For each step of the training, the model will be dealt a new batch.

By using `ImageDataGenerator`, the model's validation accuracy is higher and more consistent. This means that our model is no longer overfitting, generalizes better, make better predictions on new data.

### ASL prediction

* Prediction

The predictions are in the format of a 24 length array. Though it looks a bit different, this is the same format as our "binarized" categorical arrays from y_train and y_test. Each element of the array is a probability between 0 and 1, representing the confidence for each category. Let's make it a little more readable. We can start by finding which element of the array represents the highest probability. This can be done with the numpy library and the [argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) function.

### ASL PC camera gesture recognition

You can use you own PC camera to achieve gesture recognition













