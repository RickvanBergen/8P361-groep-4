"""
TU/e BME Project Imaging 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, auc
import time
import os



# the size of the images in the PCAM dataset
IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')

     RESCALING_FACTOR = 1./255

     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary', shuffle=False)

     return train_gen, val_gen

# def get_model1():
#     model = Sequential()
#     # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
#     model.add(Flatten(input_shape=(96,96,3)))
#     # fully connected layer with 64 neurons and ReLU nonlinearity
#
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#
#
#     # output layer with 10 nodes (one for each class) and softmax nonlinearity
#     model.add(Dense(1, activation='softmax'))
#
#
#     # compile the model
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#     return model


def get_model1(kernel_size=(3, 3), pool_size=(4, 4), first_filters=32, second_filters=64, third_filters=128):
    # build the model
    model = Sequential()

    model.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same',
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(third_filters, kernel_size, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=pool_size))

    model.add(Conv2D(1, kernel_size, activation='sigmoid', padding='same'))
    model.add(GlobalAveragePooling2D())

    # compile the model
    model.compile(SGD(lr=0.01, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model2(kernel_size=(3, 3), pool_size=(4, 4), first_filters=32, second_filters=64):
    # build the model
    model2 = Sequential()

    model2.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same',
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model2.add(MaxPool2D(pool_size=pool_size))

    model2.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
    model2.add(MaxPool2D(pool_size=pool_size))

    model2.add(Flatten())
    model2.add(Dense(64, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))

    # compile the model
    model2.compile(SGD(lr=0.01, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy'])

    return model2

model1 = get_model1()
model2 = get_model2()

# get the data generators
train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")


# save the model and weights
model1_name = 'models/model1'
model1_filepath = model1_name + '.json'
weights1_filepath = model1_name + '_weights.hdf5'

model2_name = 'models/model2'
model2_filepath = model2_name + '.json'
weights2_filepath = model2_name + '_weights.hdf5'

model1_json = model1.to_json()  # serialize model to JSON
with open(model1_filepath, 'w') as json_file:
    json_file.write(model1_json)
    model1.save_weights(weights1_filepath)

model2_json = model2.to_json()  # serialize model to JSON
with open(model2_filepath, 'w') as json_file:
    json_file.write(model2_json)
    model2.save_weights(weights2_filepath)

# define the model checkpoint and Tensorboard callbacks
checkpoint1 = ModelCheckpoint(weights1_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard1 = TensorBoard(os.path.join('logs', model1_name))
callbacks1_list = [checkpoint1, tensorboard1]

checkpoint2 = ModelCheckpoint(weights2_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard2 = TensorBoard(os.path.join('logs', model2_name))
callbacks2_list = [checkpoint2, tensorboard2]

# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size
sta = time.time()
history1 = model1.fit_generator(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks1_list)

# train the model 2
history2 = model2.fit_generator(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks2_list)
sto = time.time()
print('Computation was {}'.format(sto-sta))

true_labels = val_gen.classes
val_gen.reset()

predict1 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)
predict2 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)

predict = np.maximum(predict1,predict2)

fpr, tpr, thresholds = roc_curve(true_labels, predict)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristics')
plt.legend(loc='lower right')
plt.show()