import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, auc
import time
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import pickle


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
                                            class_mode='binary', shuffle=False)

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=val_batch_size,
                                          class_mode='binary', shuffle=False)

    return train_gen, val_gen

# with open('models\Model13_1.json', 'r') as f:
#     model1 = model_from_json(f.read())
# with open('models\Model16.json', 'r') as f:
#     model2 = model_from_json(f.read())
# with open('models\\Model26.json', 'r') as f:
#     model3 = model_from_json(f.read())
# with open('models\\Model_28.json', 'r') as f:
#     model4 = model_from_json(f.read())
# with open('models\\model_3_dense32_lr005.json', 'r') as f:
#     model5 = model_from_json(f.read())
#
# model1.load_weights('models\Model13_1_weights.hdf5')
# model2.load_weights('models\Model16_weights.hdf5')
# model3.load_weights('models\Model26_weights.hdf5')
# model4.load_weights('models\Model_28_weights.hdf5')
# model5.load_weights('models\\model_3_dense32_lr005_weights.hdf5')


#
train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
true_labels = train_gen.classes
train_steps = train_gen.n//train_gen.batch_size
#
# train_gen.reset()
#
# predict1 = model1.predict_generator(train_gen, steps=train_steps, verbose=1)
# predict2 = model2.predict_generator(train_gen, steps=train_steps, verbose=1)
# predict3 = model3.predict_generator(train_gen, steps=train_steps, verbose=1)
# predict4 = model4.predict_generator(train_gen, steps=train_steps, verbose=1)
# predict5 = model5.predict_generator(train_gen, steps=train_steps, verbose=1)

predict1 = np.expand_dims(np.loadtxt('Main model//Predictions//model13_1.csv', delimiter=','),axis=1)
predict2 = np.expand_dims(np.loadtxt('Main model//Predictions//model21.csv', delimiter=','),axis=1)
predict3 = np.expand_dims(np.loadtxt('Main model//Predictions//model16.csv', delimiter=','),axis=1)
predict4 = np.expand_dims(np.loadtxt('Main model//Predictions//model26.csv', delimiter=','),axis=1)
predict5 = np.expand_dims(np.loadtxt('Main model//Predictions//model28.csv', delimiter=','),axis=1)

train_pred = np.concatenate((predict1,predict2,predict3,predict4,predict5),axis=1)
#RandomForestClassifier 
RF_model = RandomForestClassifier(n_estimators=200,max_depth=2,verbose=1)
RF_model.fit(train_pred,true_labels)
modelname = 'Main model\models\Combination_mod\RFc_d=2.sav'
pickle.dump(RF_model, open(modelname,'wb'))