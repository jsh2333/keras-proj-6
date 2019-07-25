# Reuters News Article Classification with Keras DNN
# July 7, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu


# import packages
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.datasets import reuters


# global constants and hyper-parameters
MY_SAMPLE = 2947
NUM_CLASS = 46
MY_NUM_WORDS = 2000
MY_EPOCH = 10
MY_BATCH = 64
MY_HIDDEN = 512
MY_DROPOUT = 0.5


    ####################
    # DATABASE SETTING #
    ####################


# there are 46 news categories in reuters DB
labels = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing',
    'money-supply','coffee','sugar','trade','reserves','ship','cotton','carcass',
    'crude','nat-gas','cpi','money-fx','interest','gnp','meal-feed','alum','oilseed',
    'gold','tin','strategic-metal','livestock','retail','ipi','iron-steel','rubber',
    'heat','jobs','lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi',
    'hog','lead']


# print shape information
def show_shape():
    print('\n== DB SHAPE INFO ==')
    print('X_train shape = ', X_train.shape)
    print('X_test shape = ', X_test.shape)
    print('Y_train shape = ', Y_train.shape)
    print('Y_test shape = ', Y_test.shape)    
    print()


# read the DB and print shape info
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words = MY_NUM_WORDS, 
        test_split = 0.3)
show_shape()


# statistics on how many articles per category in the train DB
# numpy unique is useful in this case
print('\n== TRAIN DATA CONTENT INFO ==')
unique, counts = np.unique(Y_train, return_counts = True)
for i in range(len(unique)):
    print(unique[i], labels[i], "=", counts[i])


# show the same statistics visually
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(121)
plt.hist(Y_train, bins='auto')
plt.xlabel("Classes")
plt.ylabel("Number of occurrences")
plt.title("Train data")

plt.subplot(122)
plt.hist(Y_test, bins='auto')
plt.xlabel("Classes")
plt.ylabel("Number of occurrences")
plt.title("Test data")
plt.show()


# show a sample data in its raw format
print('\n== SAMPLE ARTICLE (RAW) ==')
print("article #", MY_SAMPLE)
print("category", Y_train[MY_SAMPLE], labels[Y_train[MY_SAMPLE]])
print("number of words", len(X_train[MY_SAMPLE]))
print(X_train[MY_SAMPLE])


# python dictionary: word -> index
# zero index is not used
word_to_id = reuters.get_word_index()
print('\n== DICTIONARY INFO ==')
print("There are", len(word_to_id) + 1, "words in the dictionary.")
print('The index of "the" is', word_to_id['the'])


# python dictionary: index -> word
# this is the opposite to word_to_id dictionary
id_to_word = {}
for key, value in word_to_id.items():
    id_to_word[value] = key


# function to translate the sample review
# we use python dictionary get() function
# it returns "???" if the ID is not found
# index is subtracted by 3 to handle the first 3 special characters
        # index 0 is for padding (= filling empry space)
        # index 1 is for indicating the beginning of a review
        # index 2 is for dropped word (= out of bound)
# we use python list and join() function to concatenate the words
def decoding():
    decoded = []

    for i in X_train[MY_SAMPLE]:
        word = id_to_word.get(i - 3, "???")
        decoded.append(word)

    print('\n== SAMPLE ARTICLE (DECODED) ==')
    print(" ".join(decoded))

decoding()
print("category", Y_train[MY_SAMPLE], labels[Y_train[MY_SAMPLE]])


# we will NOT do padding (as in movie review classification)
# instead we will do tokenization for the inputs
# we get a vector (numpy array) of size MY_NUM_WORDS for each input 
# the entries are integer counts
# the resulting matrix  is very big
from keras.preprocessing.text import Tokenizer
Tok = Tokenizer(num_words = MY_NUM_WORDS)
X_train = Tok.sequences_to_matrix(X_train, mode = 'count')
X_test = Tok.sequences_to_matrix(X_test, mode = 'count')

print('\n== SAMPLE ARTICLE (TOKENIZED INPUT) ==')
sample = X_train[MY_SAMPLE]
print(*sample, sep = ' ')
print("Array size:", len(sample))
print("Sum of entries:", np.sum(sample))


# output reshaping using one-hot encoding
from keras.utils import to_categorical
Y_train = to_categorical(Y_train, NUM_CLASS)
Y_test = to_categorical(Y_test, NUM_CLASS)

print('\n== SAMPLE ARTICLE (1-HOT ENCODING OUTPUT) ==')
sample = Y_train[MY_SAMPLE]
print(sample)
print("Array size:", len(sample))

show_shape()


    ###############################
    # MODEL BUILDING AND TRAINING #
    ###############################


# build a keras sequential model of our DNN
# softmax is needed for multi-class classification
model = Sequential()
model.add(Dense(MY_HIDDEN, input_shape = (MY_NUM_WORDS,)))
model.add(Activation('relu'))
model.add(Dropout(MY_DROPOUT))
model.add(Dense(NUM_CLASS))
model.add(Activation('softmax'))
model.summary()


# prediction using the model
# shape needs to change from (2000,) to (1, 2000)
def ask_question():
    sample = X_train[MY_SAMPLE]
    sample = sample.reshape(1, sample.shape[0])
    pred = model.predict(sample, verbose = 0)
    guess = np.argmax(pred)
    answer = np.argmax(Y_train[MY_SAMPLE])

    print('\n== SAMPLE QUESTION ==')
    print("My guess for sample article:", guess, labels[guess])
    print("The answer is:", answer, labels[answer])  
    print()

ask_question()


# model training and saving
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
        metrics = ['accuracy'])
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), 
        epochs = MY_EPOCH, batch_size = MY_BATCH, verbose = 1)
model.save('chap2.h5')


    ####################
    # MODEL EVALUATION #
    ####################


# evaluate the model and calculate loss and accuracy
score = model.evaluate(X_test, Y_test, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

ask_question()
