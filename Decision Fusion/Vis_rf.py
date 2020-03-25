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

path = r'C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Decision Fusion\models\RF_model_reg.sav'
rf = pickle.load(open(path,'rb'))
print(len(rf.estimators_))