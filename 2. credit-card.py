# Credit Card Fraud Detection with Keras Autoencoder
# July 16, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import zipfile

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# global constants and hyper-parameters
RANDOM_SEED = 0
MY_SPLIT = 0.25
DIM_ENCODER = 16
EPOCHS = 10
BATCH_SIZE = 200
OPTIMIZER = 'adam'
LOSS = 'mean_squared_error'
EVAL_METRIC = 'accuracy'


# directories
DB_DIR = "./database/"
PLOTS_DIR = "./plots"
MODEL_SAVE_DIR = "./saved_models"
LOG_DIR = "./logs"



    ####################
    # DATABASE SETTING #
    ####################


def load_and_preprocess_data():
    data = pd.read_csv(DB_DIR + "creditcard.csv")

    # print some statistics of the raw DB
    print('\n== GENERAL DB INFO ==')
    print(data.info())
    print('\n== FIRST 5 DATA (RAW) ==')
    print(data.head())
    print('\n== DB STATISTICS (RAW) ==')
    print(data[['Amount', 'Class']].describe())

    print('\nTotal number of transactions', data.shape[0])
    print('Total number of fraud transactions:', data['Class'].sum())


    # drop the "Time" column (axis = 1)
    # then perform data centering of "Amopunt" column with z-score
    # we need to reshape the column to 2-dimension
    # to use fit_transform()
    scaler = StandardScaler()
    data = data.drop(['Time'], axis = 1)

    amount = data['Amount'].values
    amount = amount.reshape(-1, 1)
    data['Amount'] = scaler.fit_transform(amount)

    print('\n== FIRST 5 DATA (SCALED) ==')
    print(data.head())
    return data


def get_train_and_test_data(processed_data):
    X_train, X_test = train_test_split(processed_data, test_size = MY_SPLIT, 
            random_state = RANDOM_SEED)

    # remove all fraud data (Class = 1) from train data
    # then remove "Class" column from train data
    print(X_train.shape)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis = 1)
    print(X_train.shape)


    # remove "Class" column from test data
    # and assign it to Y_test, the true label
    print(X_test.shape)
    Y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis = 1)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, X_test, Y_test


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


class MODEL():
    def __init__(self, train_data, test_data, Y_test):
        # Defining Data Variables
        self.train_data = train_data
        self.test_data = test_data
        self.Y_test = Y_test

        #Defining the model
        self.model = self.define_model()

        # Create Directories
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)

        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)


    def define_model(self):
        dim_input = self.train_data.shape[1]
        layer_input = Input(shape = (dim_input,))
        print('Decoder input shape:', dim_input)

        # encoder: 29 -> 16 -> 8 neurons in a DNN
        layer_encoder = Dense(DIM_ENCODER, activation = "tanh")(layer_input)
        layer_encoder = Dense(int(DIM_ENCODER / 2), activation = "relu")(layer_encoder)

        # decoder: 8 -> 29 neurons in a DNN
        layer_decoder = Dense(int(DIM_ENCODER / 2), activation = 'tanh')(layer_encoder)
        layer_decoder = Dense(dim_input, activation = 'relu')(layer_decoder)

        autoencoder = Model(inputs = layer_input, outputs = layer_decoder)
        autoencoder.summary()

        return autoencoder


    def train_model(self):

        self.model.compile(optimizer=OPTIMIZER,
                      loss=LOSS,
                      metrics=[EVAL_METRIC])

        checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, "trained_model.h5"),
                                     verbose = 0,
                                     save_best_only = True)

        log_tensorboard = TensorBoard(log_dir='./logs',
                                      histogram_freq = 0,
                                      write_graph = True,
                                      write_images = True)

        history = self.model.fit(self.train_data, self.train_data,
                             epochs = EPOCHS,
                             batch_size = BATCH_SIZE,
                             shuffle = True,
                             validation_data = (self.test_data, self.test_data),
                             verbose = 1,
                             callbacks = [checkpoint, log_tensorboard]).history

        self.history = history
        print("Training Done. Plotting Loss Curves")
        self.plot_loss_curves()



    def plot_loss_curves(self):
        fig = plt.figure(num = "Loss Curves")
        fig.set_size_inches(12, 6)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('Loss By Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch Num')
        plt.legend(['Train_Data', 'Test_Data'], loc='upper right');
        plt.grid(True, alpha=.25)
        plt.tight_layout()

        image_name = 'Loss_Curves.png'
        fig.savefig(os.path.join(PLOTS_DIR, image_name), dpi=fig.dpi)

        # clear the plot
        plt.clf()



    def get_trained_model(self):
        self.model = load_model(os.path.join(MODEL_SAVE_DIR, "trained_model.h5"))

    def get_test_predictions(self):
        self.test_predictions = self.model.predict(self.test_data)

    def plot_reconstruction_error_by_class(self):
        self.get_test_predictions()
        mse = np.mean(np.power(self.test_data - self.test_predictions, 2), axis=1)
        self.recon_error = pd.DataFrame({'recon_error': mse,
                                 'true_class': self.Y_test})

        ## Plotting the errors by class
        # Normal Transactions
        fig = plt.figure(num = "Recon Error with Normal Transactions")
        fig.set_size_inches(12, 6)
        ax = fig.add_subplot(111)
        normal_error_df = self.recon_error[(self.recon_error['true_class'] == 0) & (self.recon_error['recon_error'] < 50)]
        _ = ax.hist(normal_error_df.recon_error.values, bins=20)
        plt.xlabel("Recon Error Bins")
        plt.ylabel("Num Samples")
        plt.title("Recon Error with Normal Transactions")
        plt.tight_layout()
        image_name = "Recon_Error_with_Normal_Transactions.png"
        fig.savefig(os.path.join(PLOTS_DIR, image_name), dpi=fig.dpi)
        plt.clf()

        # Fraud Transactions
        fig = plt.figure(num="Recon Error with Fraud Transactions")
        fig.set_size_inches(12, 6)
        ax = fig.add_subplot(111)
        fraud_error_df = self.recon_error[(self.recon_error['true_class'] == 1)]
        _ = ax.hist(fraud_error_df.recon_error.values, bins=20)
        plt.xlabel("Recon Error Bins")
        plt.ylabel("Num Samples")
        plt.title("Recon Error with Fraud Transactions")
        plt.tight_layout()
        image_name = "Recon_Error_with_Fraud_Transactions.png"
        fig.savefig(os.path.join(PLOTS_DIR, image_name), dpi=fig.dpi)
        plt.clf()

    def get_precision_recall_curves(self):
        precision, recall, threshold = precision_recall_curve(self.recon_error.true_class, self.recon_error.recon_error)
        # Plotting the precision curve
        fig = plt.figure(num ="Precision Curve")
        fig.set_size_inches(12, 6)

        plt.plot(threshold, precision[1:], 'g', label='Precision curve')
        plt.title('Precision By Recon Error Threshold Values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.xlim(0,200)
        plt.tight_layout()
        image_name = 'Precision_Threshold_Curve.png'
        fig.savefig(os.path.join(PLOTS_DIR, image_name), dpi=fig.dpi)
        plt.clf()

        plt.plot(threshold, recall[1:], 'g', label='Recall curve')
        plt.title('Recall By Recon Error Threshold Values')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.tight_layout()
        image_name = 'Recall_Threshold_Curve.png'
        fig.savefig(os.path.join(PLOTS_DIR, image_name), dpi=fig.dpi)
        plt.clf()

    def get_confusion_matrix(self, min_recall = 0.8):
        # Get the confusion matrix with min desired recall on the testing dataset used.
        precision, recall, threshold = precision_recall_curve(self.recon_error.true_class, self.recon_error.recon_error)
        idx = list(filter(lambda x: x[1] > min_recall, enumerate(recall[1:])))[-1][0]
        th = threshold[idx]
        print ("Min recall is : %f, Threshold for recon error is: %f " %(recall[idx+1], th))

        # Get the confusion matrix
        predicted_class = [1 if e > th else 0 for e in self.recon_error.recon_error.values]
        cnf_matrix = confusion_matrix(self.recon_error.true_class, predicted_class)
        classes = ['Normal','Fraud']

        fig = plt.figure(figsize=(12, 12))
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        image_name = 'Confusion_Matrix_with_threshold_{}.png'.format(th)
        fig.savefig(os.path.join(PLOTS_DIR, image_name), dpi=fig.dpi)
        plt.clf()


    ####################
    # MODEL EVALUATION #
    ####################


# Loading and processing data
print ("Loading the data")
processed_data = load_and_preprocess_data()
print ("Getting train and test dataset")
X_train, X_test, Y_test = get_train_and_test_data(processed_data)

model_obj = MODEL(X_train, X_test, Y_test)

print ("Training the model")
model_obj.train_model()


print ("Loading the trained model")
model_obj.get_trained_model()
print ("Get Reconstruction Loss By Class")
model_obj.plot_reconstruction_error_by_class()
print ("Getting Precision Recall Curves by Thresholds")
model_obj.get_precision_recall_curves()
print ("Get confusion matrix with 80% recall on Test Dataset")
model_obj.get_confusion_matrix(min_recall = 0.8)

