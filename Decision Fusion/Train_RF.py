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
from sklearn.ensemble import RandomForestClassifier
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

with open('models\model1.json', 'r') as f:
    model1 = model_from_json(f.read())
with open('models\model2.json', 'r') as f:
    model2 = model_from_json(f.read())
with open('models\\transfer_model.json', 'r') as f:
    model3 = model_from_json(f.read())
model1.load_weights('models\model1_weights.hdf5')
model2.load_weights('models\model2_weights.hdf5')
model3.load_weights('models\\transfer_model_weights.hdf5')



train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
true_labels = train_gen.classes
train_steps = train_gen.n//train_gen.batch_size

train_gen.reset()

predict1 = model1.predict_generator(train_gen, steps=train_steps, verbose=1)
predict2 = model2.predict_generator(train_gen, steps=train_steps, verbose=1)
predict3 = model3.predict_generator(train_gen, steps=train_steps, verbose=1)

train_pred = np.concatenate((predict1,predict2,predict3),axis=1)

RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(train_pred,true_labels)
modelname = 'models\RF_model.sav'
pickle.dump(RF_model, open(modelname,'wb'))