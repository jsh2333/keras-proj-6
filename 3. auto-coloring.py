# Auto-coloring with Keras Unet
# July 18, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import matplotlib.pyplot as plt
import os
import numpy as np
from keras import models, backend
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import UpSampling2D, BatchNormalization
from keras.layers import  Concatenate, Activation
from keras import datasets, utils
from sklearn.preprocessing import minmax_scale
from keras.models import Model


# global constants and hyper-parameters
DIM = 32
RGB_CH = 3
GRAY_CH = 1
MY_EPOCH = 10
MY_BATCH = 200
OUT_DIR = "./output"
MODEL_DIR = "./model"


# create directories
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


    ####################
    # DATABASE SETTING #
    ####################


(X_train, _), (X_test, _) = datasets.cifar10.load_data()

print('\n== SHAPE INFO ==')
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)


# data scaling
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


X_train_out = X_train.reshape(X_train.shape[0], DIM, DIM, RGB_CH)
X_test_out = X_test.reshape(X_test.shape[0], DIM, DIM, RGB_CH)
input_shape = (DIM, DIM, GRAY_CH)


# convert color to gray: easy (the oopsite is hard)
# teh shape changes from (32, 32, 3) to (32, 32, 1)
# we use a special formular to make it gray
# "..." is called python ellipsis
def RGB2Gray(X):
    R = X[..., 0:1]
    G = X[..., 1:2]
    B = X[..., 2:3]

    return 0.299 * R + 0.587 * G + 0.114 * B


X_train_in = RGB2Gray(X_train_out)
X_test_in = RGB2Gray(X_test_out)

print(X_train[0].shape)
plt.imshow(X_train[0])
#plt.show()

print(X_train_out[0].shape)
plt.imshow(X_train_out[0])
#plt.show()

print(X_train_in[0].shape)
plt.imshow(X_train_in[0].reshape(DIM, DIM))
plt.gray()
#plt.show()
plt.clf()

print('X_train input:', X_train_in.shape)
print('X_train output:', X_train_out.shape)
print('X_test input:', X_test_in.shape)        
print('X_test output:', X_test_out.shape)




    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# mp_flag decides if we do max pooling or not
def conv(x, ch_out, mp_flag = True):
    x = MaxPooling2D((2, 2), padding = 'same')(x) if mp_flag else x
    x = Conv2D(ch_out, (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Dropout(0.05)(x)
    x = Conv2D(ch_out, (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    return x


# axis = 3 is for RGB channel
def deconv_unet(x, ext, ch_out):
    x = UpSampling2D((2, 2))(x)

    # concatenation makes this ANN a UNET
    # it adds non-neighboring synaptic connections
    # between ext (= old layer) and x (= currunt layer) 
    x = Concatenate(axis = 3)([x, ext])

    x = Conv2D(ch_out, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    x = Conv2D(ch_out, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    return x


# input
original = Input(shape = input_shape)
        

# encoder construction
c1 = conv(original, 16, mp_flag = False)
print('\n== SHAPE INFO DURING UNET CONSTRUCTION ==')          
print('Input shape to Unet:', input_shape)
print('Shape after the first CNN:', c1.shape)
c2 = conv(c1, 32)
print('Shape after the second CNN:', c2.shape)
encoded = conv(c2, 64)
print('Shape after the third CNN:', encoded.shape)


# decoder construction
# connect c2 layer to the new layer
x = deconv_unet(encoded, c2, 32)
print('\nShape after the first de-CNN:', x.shape)

# connect c1 layer to the new layer
x = deconv_unet(x, c1, 16)
print('Shape after the second de-CNN:', x.shape)

decoded = Conv2D(RGB_CH, (3, 3), activation = 'sigmoid', padding = 'same')(x)
print('Shape after the final CNN:', decoded.shape)

unet = Model(inputs = original, outputs = decoded)
unet.summary()


unet.compile(optimizer = 'adadelta', loss = 'mse')
history = unet.fit(X_train_in, X_train_out,
                    epochs = MY_EPOCH,
                    batch_size = MY_BATCH,
                    shuffle = True,
                    validation_split = 0.2)

unet.save(os.path.join(MODEL_DIR, 'chap3.h5'))


    ####################
    # MODEL EVALUATION #
    ####################

def plot_loss(history):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)

    plt.savefig(os.path.join(OUT_DIR, 'chap3-plot.png'))
    print('\n== LOSS PLOT SAVED ==')
    plt.clf()


def show_images(X_test_in):

    # color in, color out
    decoded_imgs_org = unet.predict(X_test_in)
    decoded_imgs = decoded_imgs_org


    # shape changes from (10000, 32, 32, 1) to (10000, 32, 32)
    print(X_test_in.shape)
    X_test_in = X_test_in[..., 0]
    print(X_test_in.shape)

    n = 10
    plt.figure(figsize = (20, 6))

    for i in range(n):
        # gray-scale image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test_in[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # auto-colored image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # golden truth image
        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(X_test_out[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.savefig(os.path.join(OUT_DIR, 'chap3-sample.png'))
    print('\n== SAMPLE COLORING RESULTS SAVED ==')
    plt.clf()


plot_loss(history)
show_images(X_test_in)
