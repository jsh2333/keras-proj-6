# Stick Price Prediction with Keras RNN
# July 5, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense


# global constants and hyper-parameters
TIME_STEP = 3
MY_NUM_NEURON = 256
MY_CUTOFF = 0.7
MY_EPOCH = 10
MY_BATCH = 64


    ####################
    # DATABASE SETTING #
    ####################


# load amazon stock price history
# header = 0 means the header is in row 0
# we only use date and closing price
raw_DB = pd.read_csv('amazon.csv', header = 0, usecols = ['Date', 'Close'],
        parse_dates = True, index_col = 'Date')


# print some statistics of the raw DB
print('\n== GENERAL DB INFO ==')
print(raw_DB.info())
print('\n== FIRST 5 DATA (RAW) ==')
print(raw_DB.head())
print('\n== DB STATISTICS (RAW) ==')
print(raw_DB.describe())


# show visual trend
plt.figure(figsize = (10,5))
plt.plot(raw_DB)
plt.show()


# display the logarithm of returns
change = raw_DB.pct_change()
log_return = np.log(1 + change) 
print('\n== LOG-RETURN OF THE LAST 10 DAYS ==')
print(log_return.tail(10))

plt.figure(figsize = (10,5))
plt.plot(log_return)
plt.show()


# scaling with z-score: z = (x - u) / s
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_DB = scaler.fit_transform(raw_DB)


# collect scaled DB stats using described()
# pandas framing is needed 
# because scaling returns numpy array
summary = pd.DataFrame(scaled_DB, columns = ['Close'])
print('\n== FIRST 5 DATA (SCALED) ==')
print(summary.head())
print('\n== DB STATISTICS (SCALED) ==')
print(summary.describe())


# split the DB into training (70%) and test (30%) sets
cutoff = int(len(scaled_DB) * MY_CUTOFF)
TrainData = scaled_DB[0 : cutoff, :]
TestData = scaled_DB[cutoff : len(scaled_DB), :]


# TIME_STEP: how many old data to be used to predict new
# input dataset is 2-dim, and the data is in the first dim
# we must return numpy arrays
def split_DB(dataset):
    input, output = [], []
    
    for i in range(len(dataset)- TIME_STEP):
            a = dataset[i:(i+ TIME_STEP), 0]
            input.append(a)
            output.append(dataset[i + TIME_STEP, 0])

    return np.array(input), np.array(output)

X_train, Y_train = split_DB(TrainData)
X_test, Y_test = split_DB(TestData)

range = 5
print('\n== FIRST FEW DATA (SCALED) ==')
print(scaled_DB[0:range])
print('\n== FIRST FEW TRAIN DATA (INPUT) ==')
print(X_train[0:range])
print('\n== FIRST FEW TRAIN DATA (OUTPUT) ==')
print(Y_train[0:range])


# input reshaping for LSTM
# LSTM input accepts 3D array: (batch_size, time_steps, seq_len)
X_train = np.reshape(X_train, (X_train.shape[0], TIME_STEP, 1))
X_test = np.reshape(X_test, (X_test.shape[0], TIME_STEP, 1))

print('\n== SHAPE INFO (AFTER RESHAPING) ==')
print('X train shape = ', X_train.shape)
print('X test shape = ', X_test.shape)


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build a keras sequential model of our RNN
# input_shape needs 2D: (time_steps, seq_len)
# we do not use embedding as we are not doing natual language process (NLP)
# output neuron count is MY_NUM_NEURON
model = Sequential()
model.add(LSTM(MY_NUM_NEURON, input_shape = (TIME_STEP, 1)))
model.add(Dense(1))
model.summary()


# model training and saving
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, batch_size = MY_BATCH, verbose = 1)
model.save('chap1.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# model evaluation
score = model.evaluate(X_train, Y_train, verbose = 1)
print('Keras Model Loss =', score[0])
print('Keras Model Accuracy =', score[1])


# model prediction
TrainPred = model.predict(X_train)
TestPred = model.predict(X_test)


# reverse z-score data back to the original
TrainPred = scaler.inverse_transform(TrainPred)
TestPred = scaler.inverse_transform(TestPred)


# reshaping test prediction for plotting
# empty_like: return a new array with 
# the same shape and type as a given array
# we initialize with "nan" and fill-up with test data prediction
TestPredictPlot = np.empty_like(scaled_DB)
TestPredictPlot[:, :] = np.nan
TestPredictPlot[len(scaled_DB) - len(TestPred) : len(scaled_DB), :] = TestPred


# blue: original data
# orange: prediction using train data
# green: prediction using test data
plt.plot(scaler.inverse_transform(scaled_DB))
plt.plot(TrainPred)
plt.plot(TestPredictPlot)
plt.show()
