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
from sklearn import svm
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
                                            class_mode='binary', shuffle=False)

    val_gen = datagen.flow_from_directory(valid_path,
                                          target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                          batch_size=val_batch_size,
                                          class_mode='binary', shuffle=False)

    return train_gen, val_gen

# model1 = ut.load_pr_model('model1')
# model2 = ut.load_pr_model('model2')
# model3 = ut.load_pr_model('transfer_model')
#
#
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
predict1 = np.expand_dims(np.loadtxt('Main model//Predictions//model13_1.csv', delimiter=','),axis=1)
predict2 = np.expand_dims(np.loadtxt('Main_model//Predictions//model21.csv', delimiter=','),axis=1)
predict3 = np.expand_dims(np.loadtxt('Main model//Predictions//model16.csv', delimiter=','),axis=1)
predict4 = np.expand_dims(np.loadtxt('Main model//Predictions//model26.csv', delimiter=','),axis=1)
predict5 = np.expand_dims(np.loadtxt('Main model//Predictions//model28.csv', delimiter=','),axis=1)
# predict3 = np.expand_dims(np.loadtxt('Predictions//transfer_model.csv', delimiter=','),axis=1)
# predict4 = np.expand_dims(np.loadtxt('Predictions//4_Model_2+conv(128).csv', delimiter=','),axis=1)
# predict5 = np.expand_dims(np.loadtxt('Predictions//13_Model_12+lr(0.001).csv', delimiter=','),axis=1)
# predict98 = np.expand_dims(np.loadtxt('Predictions//Model31.csv', delimiter=','),axis=1)


train_pred3 = np.concatenate((predict1,predict2,predict3),axis=1)


# SVM_model = svm.SVC(kernel='linear',C=1, verbose=1, gamma='scale', max_iter=10000, probability=True)
# SVM_model.fit(train_pred3,true_labels)
# modelname = 'Main model\\models\Combination_mod\\SVC.sav'
# pickle.dump(SVM_model, open(modelname,'wb'))
#
# SVM_model = svm.SVC(kernel='linear',C=1, verbose=1, gamma='scale', max_iter=100000, probability=True)
# SVM_model.fit(train_pred3,true_labels)
# modelname = 'Main model\\models\Combination_mod\\SVC_iter=100000.sav'
# pickle.dump(SVM_model, open(modelname,'wb'))


SVM_model = svm.SVC(kernel='rbf',C=0.0001, verbose=1, gamma='scale', max_iter=10000, probability=True)
SVM_model.fit(train_pred3,true_labels)
modelname = 'Main model\\models\Combination_mod\\SVC_rbf_C=0.0001.sav'
pickle.dump(SVM_model, open(modelname,'wb'))
# SVM_model = svm.SVC(kernel='poly', degree = 2, C=1, verbose=1, gamma='scale', max_iter=10000, probability=True)
# SVM_model.fit(train_pred3,true_labels)
# modelname = 'Main model\\models\Combination_mod\\SVC_poly=2.sav'
# pickle.dump(SVM_model, open(modelname,'wb'))
#
# SVM_model = svm.SVC(kernel='poly', degree = 3, C=1, verbose=1, gamma='scale', max_iter=10000, probability=True)
# SVM_model.fit(train_pred3,true_labels)
# modelname = 'Main model\\models\Combination_mod\\SVC_poly=3.sav'
# pickle.dump(SVM_model, open(modelname,'wb'))

# SVM_model = svm.SVC(kernel='poly', degree=5 ,C=1, verbose=1, gamma='scale', max_iter=10000, probability=True)
# SVM_model.fit(train_pred3,true_labels)
# modelname = 'models\Combination_mod\\SVC_poly5.sav'
# pickle.dump(SVM_model, open(modelname,'wb'))