"""
TU/e BME Project Imaging 2019
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import TensorBoard


# load the dataset using the builtin Keras method
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# derive a validation set from the training set
# the original training set is split into 
# new training set (90%) and a validation set (10%)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)


# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxH, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))


# convert the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')


# normalize our data values to the range [0,1]
X_train /= 255
X_val /= 255
X_test /= 255

def DefineType(item):
    if item in [1, 7]:
        output = 0  # vertical digit
    if item in [0, 6, 8, 9]:
        output = 1  # loopy digit
    if item in [2, 5]:
        output = 2  # curly digit
    if item in [3, 4]:
        output = 3  # other
    return output

y_train = np.array(list(map(DefineType, y_train)))
y_val = np.array(list(map(DefineType, y_val)))
y_test = np.array(list(map(DefineType, y_test)))

# convert 1D class arrays to 4D class matrices
y_train = np_utils.to_categorical(y_train, 4)
y_val = np_utils.to_categorical(y_val, 4)
y_test = np_utils.to_categorical(y_test, 4)

model = Sequential()
# flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
model.add(Flatten(input_shape=(28,28,1))) 
# fully connected layer with 64 neurons and ReLU nonlinearity
# model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))


# output layer with 10 nodes (one for each class) and softmax nonlinearity
model.add(Dense(4, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# use this variable to name your model
# model_name="{numlay}layer64nod4out".format(numlay = len(model.layers))
model_name = '2layer_(128,64)node_4out'

print(model_name)
# create a way to monitor our model in Tensorboard
tensorboard = TensorBoard("logs3\{}".format(model_name))

# train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])


score = model.evaluate(X_test, y_test, verbose=0)


print("Loss: ",score[0])
print("Accuracy: ",score[1])
