import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential, Model, model_from_json
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import concatenate as kconc
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, auc
import time
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import pickle
import Util as ut


IMAGE_SIZE = 96


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

    # dataset parameters
    train_path = os.path.join(base_dir, 'train+val', 'train')
    valid_path = os.path.join(base_dir, 'train+val', 'valid')

    RESCALING_FACTOR = 1. / 255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(train_path,
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size=train_batch_size,
                                            class_mode='binary', shuffle=True)

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=val_batch_size,
                                          class_mode='binary', shuffle=True)
    true_labels_train = train_gen.classes
    true_labels_val = val_gen.classes
    train_steps = train_gen.n // train_gen.batch_size
    val_steps = val_gen.n // val_gen.batch_size

    train_gen.reset()
    model1 = ut.load_pr_model('model1')
    model2 = ut.load_pr_model('model2')
    model3 = ut.load_pr_model('transfer_model')
    pred_tr1 = model2.predict_generator(train_gen, steps=train_steps, verbose=1)
    pred_tr2 = model2.predict_generator(train_gen, steps=train_steps, verbose=1)
    pred_tr3 = model3.predict_generator(train_gen, steps=train_steps, verbose=1)
    pred_v1 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)
    pred_v2 = model2.predict_generator(val_gen, steps=val_steps, verbose=1)
    pred_v3 = model3.predict_generator(val_gen, steps=val_steps, verbose=1)
    return train_gen, val_gen, pred_tr1, pred_tr2, pred_tr3, pred_v1, pred_v2, pred_v3

train_gen, val_gen, pred_tr1, pred_tr2, pred_tr3, pred_v1, pred_v2, pred_v3 = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")

# train_gen, val_gen = get_pcam_generators(
#     r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")


# model1 = ut.load_pr_model('model1')
# model2 = ut.load_pr_model('model2')
# model3 = ut.load_pr_model('transfer_model')

true_labels_train = train_gen.classes
true_labels_val = val_gen.classes
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size
#
# train_gen.reset()
#
# pred_v1 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)
# pred_v2 = model2.predict_generator(val_gen, steps=val_steps, verbose=1)
# pred_v3 = model3.predict_generator(val_gen, steps=val_steps, verbose=1)

# pred_tr1 = np.expand_dims(np.loadtxt('Predictions//model1.csv', delimiter=','),axis=1)
# pred_tr2 = np.expand_dims(np.loadtxt('Predictions//model2.csv', delimiter=','),axis=1)
# pred_tr3 = np.expand_dims(np.loadtxt('Predictions//transfer_model.csv', delimiter=','),axis=1)

print(pred_v1.shape)
print(true_labels_val.shape)
print(pred_tr1.shape)
print(true_labels_train.shape)

# def get_modelcomb():
#     # build the model
#     in1 = Input(shape=(1, ))
#     in2 = Input(shape=(1, ))
#     in3 = Input(shape=(1, ))
#     x = kconc([in1, in2, in3])
#     x = (Dense(3, input_dim=3, activation='relu'))(x)
#     x = Dense(100, activation='relu')(x)
#     x = Dense(5, activation='relu')(x)
#     output = Dense(1, activation='softmax', name='output')(x)
#
#     model = Model(inputs=[in1, in2, in3], outputs=[output])
#
#     model.compile(SGD(lr=0.01, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy'])
#
#     return model

def get_modelcomb():
    # build the model
    in1 = Input(shape=(1, ))
    x = (Dense(3, input_dim=3, activation='relu'))(in1)
    # x = Dense(100, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    output = Dense(1, activation='softmax', name='output')(x)

    model = Model(inputs=in1, outputs=output)

    model.compile(SGD(lr=0.01, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy'])

    return model

modelcomb = get_modelcomb()
callbacks_list = ut.save_com_model('modelcombi1in',modelcomb)
modelcomb.fit(pred_tr2,true_labels_train, batch_size=256, epochs=10, verbose=1, validation_data=(pred_v2,true_labels_val), shuffle= True ,callbacks=callbacks_list)




# modelcomb.fit([pred_tr1,pred_tr2,pred_tr3],true_labels_train, batch_size=64, epochs=10, verbose=1, validation_data=([pred_v1, pred_v2, pred_v3],true_labels_val), callbacks=callbacks_list)

# history = modelcomb.fit_generator(([pred_tr1,pred_tr2,pred_tr3],true_labels_train),
#                                   steps_per_epoch=train_steps,
#                                   epochs=10,
#                                   verbose=1,
#                                   validation_data=([pred_v1, pred_v2, pred_v3],true_labels_val),
#                                   validation_steps=val_steps,
#                                   callbacks=callbacks_list)

