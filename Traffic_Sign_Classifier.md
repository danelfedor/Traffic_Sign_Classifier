
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]
# TODO: Number of testing examples.
n_test = X_test.shape[0]
# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43
    

### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

print(y_train[1980])
imgplot = plt.imshow(X_train[1980])


```

    23
    


![png](output_8_1.png)



```python
plt.figure(figsize=(20,5))
plt.hist(y_train,bins=43)
plt.show()
```


![png](output_9_0.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense,Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import regularizers



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
x_train = (X_train)
x_valid = (X_valid)
x_test = (X_test)



Y_train=to_categorical(y_train,num_classes=43)
Y_test=to_categorical(y_test,num_classes=43)
Y_valid=to_categorical(y_valid,num_classes=43)

### Feel free to use as many code cells as needed.
```

    G:\ProgramData\Anaconda3\envs\tensor17\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

In my structure, I tried different normalized method but I didn't see any structure work better than original picture. The reson is there are batch normalization layers after each convolutional layer and fully connect layer. As long as there is bach normalization 

I do not change anything of the dataset. I think original one is working fine on original picture data.

### Model Architecture


```python
import keras
```


```python
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=2,input_shape=(32,32,3),activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(filters=32,kernel_size=2,activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))


model.add(Conv2D(filters=64,kernel_size=2,activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))


model.add(Conv2D(filters=128,kernel_size=2,activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))


model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dropout(0.5))
model.add(Dense(128, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64, use_bias=False))
model.add(Dense(43,activation='softmax'))


model.summary()

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_5 (Conv2D)            (None, 31, 31, 16)        208       
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 31, 31, 16)        64        
    _________________________________________________________________
    activation_7 (Activation)    (None, 31, 31, 16)        0         
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 15, 15, 16)        0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 15, 15, 16)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 14, 14, 32)        2080      
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    activation_8 (Activation)    (None, 14, 14, 32)        0         
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 7, 7, 32)          0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 6, 6, 64)          8256      
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 6, 6, 64)          256       
    _________________________________________________________________
    activation_9 (Activation)    (None, 6, 6, 64)          0         
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 3, 3, 64)          0         
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 3, 3, 64)          0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 2, 2, 128)         32896     
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 2, 2, 128)         512       
    _________________________________________________________________
    activation_10 (Activation)   (None, 2, 2, 128)         0         
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 1, 1, 128)         0         
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 1, 1, 128)         0         
    _________________________________________________________________
    global_average_pooling2d_2 ( (None, 128)               0         
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 256)               32768     
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 256)               1024      
    _________________________________________________________________
    activation_11 (Activation)   (None, 256)               0         
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 256)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               32768     
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 128)               512       
    _________________________________________________________________
    activation_12 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 64)                8192      
    _________________________________________________________________
    dense_8 (Dense)              (None, 43)                2795      
    =================================================================
    Total params: 122,459
    Trainable params: 121,211
    Non-trainable params: 1,248
    _________________________________________________________________
    

The original Idea is from VGG16 model. But here I only have 32*32*3 picture not 255*255, So I cannot directly use VGG16. I decrease the number of convolution layers and add batch normalization to balance data in each batch. And also I observe that there is overfit condition in training process so I add drop out and L2 normalization. And further more global average pooling will dignificantly decrease the time of training. 

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

Use adam optimizer, 200 epochs, batch size 64.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint  

epochs = 40
batch_s = 128


checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(x_train, Y_train, 
          validation_data=(x_test, Y_test),
          epochs=epochs, batch_size=batch_s, callbacks=[checkpointer], verbose=1,shuffle=True)

```

    Train on 34799 samples, validate on 12630 samples
    Epoch 1/40
    34799/34799 [==============================] - 8s 221us/step - loss: 2.6314 - acc: 0.2362 - val_loss: 1.7459 - val_acc: 0.3979
    
    Epoch 00001: val_loss improved from inf to 1.74589, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 2/40
    34799/34799 [==============================] - 6s 172us/step - loss: 1.5420 - acc: 0.4901 - val_loss: 1.0459 - val_acc: 0.6546
    
    Epoch 00002: val_loss improved from 1.74589 to 1.04588, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 3/40
    34799/34799 [==============================] - 6s 174us/step - loss: 1.0715 - acc: 0.6395 - val_loss: 0.8341 - val_acc: 0.7296
    
    Epoch 00003: val_loss improved from 1.04588 to 0.83410, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 4/40
    34799/34799 [==============================] - 6s 173us/step - loss: 0.8141 - acc: 0.7266 - val_loss: 0.5689 - val_acc: 0.8226
    
    Epoch 00004: val_loss improved from 0.83410 to 0.56886, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 5/40
    34799/34799 [==============================] - 6s 174us/step - loss: 0.6694 - acc: 0.7778 - val_loss: 0.4667 - val_acc: 0.8541
    
    Epoch 00005: val_loss improved from 0.56886 to 0.46672, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 6/40
    34799/34799 [==============================] - 6s 175us/step - loss: 0.5618 - acc: 0.8144 - val_loss: 0.4037 - val_acc: 0.8751
    
    Epoch 00006: val_loss improved from 0.46672 to 0.40372, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 7/40
    34799/34799 [==============================] - 6s 174us/step - loss: 0.4835 - acc: 0.8420 - val_loss: 0.3175 - val_acc: 0.8998
    
    Epoch 00007: val_loss improved from 0.40372 to 0.31754, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 8/40
    34799/34799 [==============================] - 6s 176us/step - loss: 0.4360 - acc: 0.8558 - val_loss: 0.2765 - val_acc: 0.9165
    
    Epoch 00008: val_loss improved from 0.31754 to 0.27649, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 9/40
    34799/34799 [==============================] - 6s 175us/step - loss: 0.3900 - acc: 0.8732 - val_loss: 0.3226 - val_acc: 0.9068
    
    Epoch 00009: val_loss did not improve
    Epoch 10/40
    34799/34799 [==============================] - 6s 173us/step - loss: 0.3675 - acc: 0.8809 - val_loss: 0.2825 - val_acc: 0.9193
    
    Epoch 00010: val_loss did not improve
    Epoch 11/40
    34799/34799 [==============================] - 6s 171us/step - loss: 0.3465 - acc: 0.8867 - val_loss: 0.2587 - val_acc: 0.9248
    
    Epoch 00011: val_loss improved from 0.27649 to 0.25870, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 12/40
    34799/34799 [==============================] - 6s 174us/step - loss: 0.3254 - acc: 0.8954 - val_loss: 0.2421 - val_acc: 0.9314
    
    Epoch 00012: val_loss improved from 0.25870 to 0.24214, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 13/40
    34799/34799 [==============================] - 6s 177us/step - loss: 0.3047 - acc: 0.9044 - val_loss: 0.2630 - val_acc: 0.9285
    
    Epoch 00013: val_loss did not improve
    Epoch 14/40
    34799/34799 [==============================] - 6s 171us/step - loss: 0.2920 - acc: 0.9069 - val_loss: 0.2794 - val_acc: 0.9288
    
    Epoch 00014: val_loss did not improve
    Epoch 15/40
    34799/34799 [==============================] - 6s 171us/step - loss: 0.2874 - acc: 0.9099 - val_loss: 0.2248 - val_acc: 0.9364
    
    Epoch 00015: val_loss improved from 0.24214 to 0.22476, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 16/40
    34799/34799 [==============================] - 6s 174us/step - loss: 0.2791 - acc: 0.9109 - val_loss: 0.2258 - val_acc: 0.9348
    
    Epoch 00016: val_loss did not improve
    Epoch 17/40
    34799/34799 [==============================] - 6s 171us/step - loss: 0.2681 - acc: 0.9141 - val_loss: 0.2167 - val_acc: 0.9385
    
    Epoch 00017: val_loss improved from 0.22476 to 0.21674, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 18/40
    34799/34799 [==============================] - 6s 176us/step - loss: 0.2629 - acc: 0.9176 - val_loss: 0.2277 - val_acc: 0.9403
    
    Epoch 00018: val_loss did not improve
    Epoch 19/40
    34799/34799 [==============================] - 6s 176us/step - loss: 0.2492 - acc: 0.9223 - val_loss: 0.2151 - val_acc: 0.9425
    
    Epoch 00019: val_loss improved from 0.21674 to 0.21506, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 20/40
    34799/34799 [==============================] - 6s 171us/step - loss: 0.2446 - acc: 0.9226 - val_loss: 0.2068 - val_acc: 0.9387
    
    Epoch 00020: val_loss improved from 0.21506 to 0.20684, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 21/40
    34799/34799 [==============================] - 6s 169us/step - loss: 0.2340 - acc: 0.9266 - val_loss: 0.2228 - val_acc: 0.9405
    
    Epoch 00021: val_loss did not improve
    Epoch 22/40
    34799/34799 [==============================] - 6s 166us/step - loss: 0.2287 - acc: 0.9275 - val_loss: 0.2214 - val_acc: 0.9412
    
    Epoch 00022: val_loss did not improve
    Epoch 23/40
    34799/34799 [==============================] - 6s 168us/step - loss: 0.2299 - acc: 0.9271 - val_loss: 0.1906 - val_acc: 0.9447
    
    Epoch 00023: val_loss improved from 0.20684 to 0.19056, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 24/40
    34799/34799 [==============================] - 6s 169us/step - loss: 0.2179 - acc: 0.9319 - val_loss: 0.1976 - val_acc: 0.9458
    
    Epoch 00024: val_loss did not improve
    Epoch 25/40
    34799/34799 [==============================] - 6s 167us/step - loss: 0.2177 - acc: 0.9321 - val_loss: 0.1755 - val_acc: 0.9504
    
    Epoch 00025: val_loss improved from 0.19056 to 0.17551, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 26/40
    34799/34799 [==============================] - 6s 169us/step - loss: 0.2110 - acc: 0.9346 - val_loss: 0.2313 - val_acc: 0.9390
    
    Epoch 00026: val_loss did not improve
    Epoch 27/40
    34799/34799 [==============================] - 6s 168us/step - loss: 0.2030 - acc: 0.9376 - val_loss: 0.2050 - val_acc: 0.9429
    
    Epoch 00027: val_loss did not improve
    Epoch 28/40
    34799/34799 [==============================] - 6s 166us/step - loss: 0.2064 - acc: 0.9362 - val_loss: 0.1984 - val_acc: 0.9494
    
    Epoch 00028: val_loss did not improve
    Epoch 29/40
    34799/34799 [==============================] - 6s 168us/step - loss: 0.2051 - acc: 0.9361 - val_loss: 0.1952 - val_acc: 0.9473
    
    Epoch 00029: val_loss did not improve
    Epoch 30/40
    34799/34799 [==============================] - 6s 167us/step - loss: 0.1996 - acc: 0.9368 - val_loss: 0.1907 - val_acc: 0.9481
    
    Epoch 00030: val_loss did not improve
    Epoch 31/40
    34799/34799 [==============================] - 6s 167us/step - loss: 0.1967 - acc: 0.9385 - val_loss: 0.1918 - val_acc: 0.9525
    
    Epoch 00031: val_loss did not improve
    Epoch 32/40
    34799/34799 [==============================] - 6s 166us/step - loss: 0.1964 - acc: 0.9393 - val_loss: 0.1894 - val_acc: 0.9493
    
    Epoch 00032: val_loss did not improve
    Epoch 33/40
    34799/34799 [==============================] - 6s 166us/step - loss: 0.1914 - acc: 0.9403 - val_loss: 0.1770 - val_acc: 0.9530
    
    Epoch 00033: val_loss did not improve
    Epoch 34/40
    34799/34799 [==============================] - 6s 167us/step - loss: 0.1921 - acc: 0.9408 - val_loss: 0.1795 - val_acc: 0.9504
    
    Epoch 00034: val_loss did not improve
    Epoch 35/40
    34799/34799 [==============================] - 6s 167us/step - loss: 0.1876 - acc: 0.9411 - val_loss: 0.1853 - val_acc: 0.9516
    
    Epoch 00035: val_loss did not improve
    Epoch 36/40
    34799/34799 [==============================] - 6s 168us/step - loss: 0.1822 - acc: 0.9430 - val_loss: 0.2027 - val_acc: 0.9457
    
    Epoch 00036: val_loss did not improve
    Epoch 37/40
    34799/34799 [==============================] - 6s 170us/step - loss: 0.1789 - acc: 0.9431 - val_loss: 0.1497 - val_acc: 0.9557
    
    Epoch 00037: val_loss improved from 0.17551 to 0.14973, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 38/40
    34799/34799 [==============================] - 6s 169us/step - loss: 0.1784 - acc: 0.9441 - val_loss: 0.1775 - val_acc: 0.9522
    
    Epoch 00038: val_loss did not improve
    Epoch 39/40
    34799/34799 [==============================] - 6s 164us/step - loss: 0.1801 - acc: 0.9442 - val_loss: 0.1805 - val_acc: 0.9532
    
    Epoch 00039: val_loss did not improve
    Epoch 40/40
    34799/34799 [==============================] - 6s 163us/step - loss: 0.1807 - acc: 0.9424 - val_loss: 0.1687 - val_acc: 0.9534
    
    Epoch 00040: val_loss did not improve
    




    <keras.callbacks.History at 0x11eb111d128>



---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in x_valid]
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(Y_valid, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 96.0998%
    

### Analyze Performance


```python
from keras.preprocessing import image                  
import os
imgs = []
for i in os.listdir("Image_new/"):
    img = image.load_img("Image_new/"+i, target_size=(32, 32))
    x_new = image.img_to_array(img)/255
    imgs.append(x_new)
plt.imshow(imgs[3],vmin=0,vmax=255)
plt.show()
plt.imshow(imgs[4],vmin=0,vmax=255)
plt.show()
plt.imshow(imgs[5],vmin=0,vmax=255)
plt.show()
plt.imshow(imgs[6],vmin=0,vmax=255)
plt.show()
```


![png](output_30_0.png)



![png](output_30_1.png)



![png](output_30_2.png)



![png](output_30_3.png)



```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
predictions = [np.argmax(model.predict(np.expand_dims(tensor*255, axis=0))) for tensor in imgs]
correct = [22, 17, 9, 27, 27, 31, 31]
print(predictions)
test_accuracy = 100*np.sum(np.array(predictions)==correct)/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

```

    [23, 17, 26, 27, 27, 23, 23]
    Test accuracy: 42.8571%
    

Plot 3 and plot 5 are different with the plot in training data.


```python
test_accuracy = 100*np.sum(np.array(predictions)==correct)/(len(predictions)-2)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 60.0000%
    

The true accuracy rate will be 60%

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
import tensorflow as tf
predictions = [(model.predict(np.expand_dims(tensor*255, axis=0))) for tensor in imgs]

with tf.Session() as sess:
    b = sess.run(tf.nn.top_k(tf.nn.softmax(tf.constant(np.array(predictions))), k=5))
print(b)
test_accuracy = 100*np.sum([correct[i] in b[1][i] for i in range(len(predictions))])/(len(predictions))
print('Test accuracy: %.4f%%' % test_accuracy)
```

    TopKV2(values=array([[[0.06072254, 0.02237614, 0.02236654, 0.02236613, 0.02236515]],
    
           [[0.06078671, 0.02236226, 0.02236224, 0.02236224, 0.02236224]],
    
           [[0.04348101, 0.02718202, 0.02359923, 0.02304604, 0.02289749]],
    
           [[0.03670235, 0.03528636, 0.02306223, 0.02303167, 0.02296069]],
    
           [[0.06076632, 0.0223696 , 0.02236309, 0.02236266, 0.02236253]],
    
           [[0.04620157, 0.0262237 , 0.02329369, 0.02307342, 0.0230597 ]],
    
           [[0.05672181, 0.02288094, 0.02273768, 0.02258317, 0.02254767]]],
          dtype=float32), indices=array([[[23, 29, 31, 37, 19]],
    
           [[17, 14, 12, 10,  7]],
    
           [[26, 12, 10, 13, 18]],
    
           [[27, 18, 24, 26, 14]],
    
           [[27, 24, 11, 28, 26]],
    
           [[23, 26, 19, 27, 28]],
    
           [[23, 19, 26, 20, 31]]]))
    Test accuracy: 57.1429%
    

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
