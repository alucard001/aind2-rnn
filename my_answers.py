import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


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
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', '-', '(', ')', '+', '-', '*', '/', '\\', '_', '[', ']', '{', '}', '\'', '"', '|', '<', '>', '?', '`', '~', '@', '#', '$', '%', '^', '&']

    for p in punctuation:
        text = text.replace(p, ' ')
    
    text = text.lower()
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for idx in range(0, len(text) - window_size):
        
        new_idx_after_calculating_step_size = idx + step_size
        
        secondIdx = new_idx_after_calculating_step_size + window_size
        item = text[new_idx_after_calculating_step_size:secondIdx]
        
        outputs_index = secondIdx
        if outputs_index < len(text):
            inputs.append(item)
            outputs.append(text[outputs_index])

    return list(inputs), list(outputs)

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
