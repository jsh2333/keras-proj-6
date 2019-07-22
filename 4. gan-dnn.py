import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
from keras.optimizers import Adam
from keras import backend as K
import os

OUT_DIR = "./output/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)



def add_decorate(x):
    """
    axis = -1 --> last dimension in an array
    """
    m = K.mean(x, axis=-1, keepdims=True)
    d = K.square(x - m)
    return K.concatenate([x, d], axis=-1)


def add_decorate_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2
    shape[1] *= 2
    return tuple(shape)

# model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))


lr = 2e-4  # 0.0002
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
n_batch = 1
ni_D = 100


def model_compile(model):
    return model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


# build GAN with the following 3 parameters:
#    1. ni_D is the number of data in each batch (100 currently)
#    2. nh_D is the number of neurons in hidden layer in discriminator (50 currently)
#    3. nh_G is the number of neurons in hidden layer in generator (50 currently)
class GAN:
    def __init__(self, ni_D, nh_D, nh_G):
        self.ni_D = ni_D
        self.nh_D = nh_D
        self.nh_G = nh_G

        self.D = self.gen_D()
        self.G = self.gen_G()
        self.GD = self.make_GD()

    # DNN discriminator definition
    # ni_D is the number of data in each batch (100 currently)
    # nh_D is the number of neurons in hidden layer in discriminator (50 currently)
    def gen_D(self):
        ni_D = self.ni_D
        nh_D = self.nh_D
        D = models.Sequential()

        #D.add(Lambda(add_decorate, output_shape=add_decorate_shape, input_shape=(ni_D,)))

        # first dense layer with 50 neurons
        # input shape is (100,)
        D.add(Dense(nh_D, activation = 'relu', input_shape = (ni_D,)))

        # second dense layer 
        D.add(Dense(nh_D, activation = 'relu'))
        
        # third dense layer
        D.add(Dense(nh_D, activation = 'relu'))

        # we need a single output for classification: real or fake
        D.add(Dense(1, activation = 'sigmoid'))

        # compile the model with common setting
        model_compile(D)

        print('\n== DISCRIMINATOR MODEL DETAILS ==')
        D.summary()
        return D


    # generator definition
    # ni_D is the number of data in each batch (100 currently)
    # nh_G is the number of neurons in hidden layer in generator (50 currently)
    def gen_G(self):
        ni_D = self.ni_D
        nh_G = self.nh_D

        G = models.Sequential()

        # we use CNN for generator
        # CNN needs channel info, so we reshape the CNN input to (100, 1)
        G.add(Reshape((ni_D, 1), input_shape = (ni_D,)))

        # first 1-dimensional CNN layer
        # kernel size is 1
        # total parameter count formula = 
        # (filter_height * filter_width * input_channels + 1) * output_channel
        # (1 * 1 * 1 + 1) * 50
        G.add(Conv1D(nh_G, 1, activation = 'relu'))

        # first 1-dimensional CNN layer
        # (1 * 1 * 50 + 1) * 50
        G.add(Conv1D(nh_G, 1, activation = 'sigmoid'))

        # first 1-dimensional CNN layer
        # (1 * 1 * 50 + 1) * 1
        # the final output is a single number
        G.add(Conv1D(1, 1))

        # flatten it to one dimension
        G.add(Flatten())

        # compile the model with common setting
        model_compile(G)

        print('\n== GENERTOR MODEL DETAILS ==')
        G.summary()
        return G


    # GAN definition
    def make_GD(self):
        G, D = self.G, self.D

        GD = models.Sequential()
        
        # add the generator
        GD.add(G)

        # add the discriminator
        GD.add(D)


        D.trainable = False
        model_compile(GD)
        D.trainable = True

        print('\n== GAN MODEL DETAILS ==')
        GD.summary()

       
        return GD


    # we train the discriminator with real and fake images combined
    # and the labels that say that real data is real
    # and the fake data is fake
    # input shape is (# batch X 2, data size)
    # output shape is (# batch X 2)
    def D_train_on_batch(self, Real, Gen):
        D = self.D

        # real and fake data merged into a single array
        # two (1, 100) arrays concatenated, we get (2, 100)
        X = np.concatenate([Real, Gen], axis = 0)


        # build the output (= label) array
        # build a single array with 1's and 0's
        # the number of 1's is the same as the number of batch
        # the number of 0's is the same as the number of batch
        # so the label array becomes [1 0]
        # 1 means real, 0 fake
        y = np.array([1] * Real.shape[0] + [0] * Gen.shape[0])
        # print('Discriminator training input and output shape:', X.shape, y.shape)
        
        # Keras train_on_batch
        # (input, label)
        D.train_on_batch(X, y)


    # we train the generator with input data 
    # and the labels that say that the data is real
    # input shape is (# batch, data size)
    # output shape is (# batch,)
    def GAN_train_on_batch(self, Z):
        GD = self.GD

        # build the true label array
        # we get [1]
        y = np.array([1] * Z.shape[0])
        # print('Generator training input and output shape:', Z.shape, y.shape)


        # Keras train_on_batch
        # (input, label)
        GD.train_on_batch(Z, y)



class data_gen:
    def __init__(self, mu, sigma, ni_D):
        # random samples with mean = mu and stddev = sigma
        self.real_sample = lambda n_batch: np.random.normal(mu, sigma, (n_batch, ni_D))

        # random samples from a uniform distribution over [0, 1).
        self.in_sample = lambda n_batch: np.random.rand(n_batch, ni_D)


class Machine:
    def __init__(self, n_batch, ni_D):
        data_mean = 4
        data_stddev = 1.25

        # number of times we train discriminator in each epoch
        self.n_iter_D = 1

        # number of times we train generator in each epoch
        self.n_iter_G = 5

        # build a random database
        # mean = 4, stddev = 1.25, 1 batch, 100 data each
        self.data = data_gen(data_mean, data_stddev, ni_D)

        print('\n== DATABASE INFO ==')
        print('Number of batches:', n_batch)
        print('Number of data in each batch:', ni_D)

        print('\nReal data:')
        print(self.data.real_sample(n_batch))
        print('\nInput data:')
        print(self.data.in_sample(n_batch))

        # build GAN with 100 data in each batch
        # 50 nuerons in discriminator dense layer
        # 50 output channels in generator CNN
        self.gan = GAN(ni_D = ni_D, nh_D = 50, nh_G = 50)
        self.n_batch = n_batch


    # discriminator training
    def train_D(self):
        gan = self.gan
        n_batch = self.n_batch
        data = self.data

        # real data for discriminator training
        # this data is generated randomly
        # shape is (epoch, 100)
        real = data.real_sample(n_batch)
        
        # input data for discriminator training
        # this data is generated randomly
        # shape is (epoch, 100)
        Z = data.in_sample(n_batch)

        # fake data generated by the generator
        # we use the input data
        # shape is (epoch, 100)
        fake = gan.G.predict(Z)

        gan.D.trainable = True
        gan.D_train_on_batch(real, fake)


    # generator training
    def train_GAN(self):
        gan = self.gan
        n_batch = self.n_batch
        data = self.data

        # input data for generator training
        # this data is generated randomly again
        Z = data.in_sample(n_batch)


        # we do not train discriminator during generator training
        gan.D.trainable = False
        gan.GAN_train_on_batch(Z)


    # run epochs
    def train_epochs(self, epochs):
        for epoch in range(epochs):
            # print ('    Epoch:', epoch)

            # number of times we train discriminator in each epoc (1 currently)
            for it in range(self.n_iter_D):
                self.train_D()

            # number of times we train generator in each epoch (5 currently)
            for it in range(self.n_iter_G):
                self.train_GAN()


    # we generate test data and use them for prediction
    # test data input shape is (# test data, data size)
    # fake data shape is (# test data, data size)
    def test(self, n_test):
        gan = self.gan
        data = self.data

        # generate test data input (random)
        # n_test is the number of test data
        Z = data.in_sample(n_test)


        # our generator produces fake data using the test input
        fake = gan.G.predict(Z)
        # print('Test data input and fake shape:', Z.shape, fake.shape)
    
        return fake, Z


    def show_hist(self, Real, Gen, Z):
        plt.hist(Real.reshape(-1), histtype = 'step', label = 'Real')
        plt.hist(Gen.reshape(-1), histtype = 'step', label = 'Generated')
        plt.hist(Z.reshape(-1), histtype = 'step', label = 'Input')
        plt.legend(loc = 0)


    def test_and_show(self, n_test):
        data = self.data

        # generate random test input and their fake data
        fake, Z = self.test(n_test)

        # generate random real data
        real = data.real_sample(n_test)

        # no need to use discriminator
        # we just compare the distribution between the real and fake
        # we want them to be similar
        self.show_hist(real, fake, Z)

        print('        Real: mean = {:.2f}'.format(np.mean(real)), 
                ', std-dev = {:.2f}'.format(np.std(real)))
        print('        Fake: mean = {:.2f}'.format(np.mean(fake)), 
                ', std-dev = {:.2f}'.format(np.std(fake)))


    # n_repeat: number of stages
    # n_epochs: number of epochs
    # n_test: number of test data
    def run(self, n_repeat, n_epochs, n_test):
        for ii in range(n_repeat):
            print('\nStage', ii, '(Epoch: {})'.format(ii * n_epochs))

            self.train_epochs(n_epochs)
            self.test_and_show(n_test)

            path = "output/chap4-img-{}".format(ii)
            plt.savefig(path)
            plt.close()


class GAN_Pure(GAN):
    def __init__(self, ni_D, nh_D, nh_G):
        '''
        Discriminator input is not added
        '''
        super().__init__(ni_D, nh_D, nh_G)

    def gen_D(self):
        ni_D = self.ni_D
        nh_D = self.nh_D
        D = models.Sequential()
        # D.add(Lambda(add_decorate, output_shape=add_decorate_shape, input_shape=(ni_D,)))
        D.add(Dense(nh_D, activation='relu', input_shape=(ni_D,)))
        D.add(Dense(nh_D, activation='relu'))
        D.add(Dense(1, activation='sigmoid'))

        model_compile(D)
        return D


class Machine_Pure(Machine):
    def __init__(self, n_batch=10, ni_D=100):
        data_mean = 4
        data_stddev = 1.25

        self.data = data_gen(data_mean, data_stddev, ni_D)
        self.gan = GAN_Pure(ni_D=ni_D, nh_D=50, nh_G=50)

        self.n_batch = n_batch
        # self.ni_D = ni_D


# build DB with a given # of batches and # of data in each batch
# next we build GAN 
machine = Machine(n_batch, ni_D)


# train GAN and evaluate
machine.run(n_repeat = 10, n_epochs = 100, n_test = 20)

