'''
TU/e BME Project Imaging 2019
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

import os

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, auc


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


def get_model(kernel_size=(3, 3), pool_size=(4, 4), first_filters=32, second_filters=64, third_filters=128):

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


# get the model
model = get_model()


# get the data generators
train_gen, val_gen = get_pcam_generators(
    "C:\\Users\\20173869\\OneDrive - TU Eindhoven\\Documents\\TUe\\Jaar 3\\Jaar 3 Q3\\8P361 Project Imaging\\")


# save the model and weights
model_name = 'my_second_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.hdf5'

model_json = model.to_json()  # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=3,
                    callbacks=callbacks_list)

# ROC analysis
true_labels = val_gen.classes
val_gen.reset()

predict = model.predict_generator(val_gen, steps=val_steps, verbose=1)

fpr, tpr, _ = roc_curve(true_labels, predict)
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
