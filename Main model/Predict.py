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
                                        class_mode='binary',
                                        shuffle=False)

    val_gen = datagen.flow_from_directory(valid_path,
                                      target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                      batch_size=val_batch_size,
                                      class_mode='binary',
                                      shuffle=False)

    return train_gen, val_gen


train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
true_labels = val_gen.classes
val_steps = val_gen.n // val_gen.batch_size
train_steps = train_gen.n // train_gen.batch_size
val_gen.reset()
def predict_train(model_name):
    model = ut.load_pr_model(model_name)
    predict = model.predict_generator(train_gen, steps=train_steps, verbose=1)
    np.savetxt(os.path.join('Main model//Predictions', model_name+'.csv'), predict, delimiter=',')

predict_train('model13_1')
predict_train('model_28')
predict_train('model_30')
