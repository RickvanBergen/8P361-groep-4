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



train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
true_labels = train_gen.classes
train_steps = train_gen.n//train_gen.batch_size

train_gen.reset()

predict1 = np.expand_dims(np.loadtxt('Predictions//model1.csv', delimiter=','),axis=1)
predict2 = np.expand_dims(np.loadtxt('Predictions//model2.csv', delimiter=','),axis=1)
predict3 = np.expand_dims(np.loadtxt('Predictions//transfer_model.csv', delimiter=','),axis=1)
predict4 = np.expand_dims(np.loadtxt('Predictions//4_Model_2+conv(128).csv', delimiter=','),axis=1)
predict5 = np.expand_dims(np.loadtxt('Predictions//13_Model_12+lr(0.001).csv', delimiter=','),axis=1)
predict98 = np.expand_dims(np.loadtxt('Predictions//Model31.csv', delimiter=','),axis=1)

# predict1 = model1.predict_generator(train_gen, steps=train_steps, verbose=1)
# predict2 = model2.predict_generator(train_gen, steps=train_steps, verbose=1)
# predict3 = model3.predict_generator(train_gen, steps=train_steps, verbose=1)

train_pred = np.concatenate((predict1,predict2,predict3,predict4,predict5),axis=1)

RF_model = RandomForestRegressor(n_estimators=500, verbose=1, max_depth=10)
RF_model.fit(train_pred,true_labels)
modelname = 'models\\Combination_mod\\RF_model_reg_restr_500_10_5in.sav'
pickle.dump(RF_model, open(modelname,'wb'))

fpr, tpr, thresholds = roc_curve(true_labels, RF_model.predict(train_pred))
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
plt.title('rf')
plt.show()