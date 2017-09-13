import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras

import string

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for idx in range(0, len(series) - window_size):
        secondIdx = idx + window_size
        item = series[idx:secondIdx]
        X.append(item)
        
        y_index = secondIdx
        if y_index < len(series):
            y.append(series[y_index])
            
    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)    
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
# https://stackoverflow.com/questions/5640630/array-filter-in-python
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    allowedChars = set(list(string.ascii_lowercase) + punctuation)
    
    for t in text:
        if t not in allowedChars:
            text = text.replace(t, ' ')
        
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
          
    lastIdx = window_size
    
    while(lastIdx < len(text)):
        inputs.append(text[lastIdx - window_size:lastIdx])
        outputs.append(text[lastIdx])
        lastIdx += step_size

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
