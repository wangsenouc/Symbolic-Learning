from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import time


def gen_data():
    Y = list()
    X = [x + 3 for x in range(-2, 43, 3)]

    for i in X:
        output_vector = list()
        output_vector.append(i + 1)
        output_vector.append(i + 2)
        output_vector.append(i + 3)
        Y.append(output_vector)

    print(X)
    print(Y)
    X = np.array(X).reshape(15, 1, 1)
    Y = np.array(Y)
    return X, Y


def main():
    X, Y = gen_data()
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=1000, validation_split=0.2, batch_size=3, verbose=0)

    test_input = array([10])
    test_input = test_input.reshape((1, 1, 1))
    test_output = model.predict(test_input, verbose=0)
    print(test_output)


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

