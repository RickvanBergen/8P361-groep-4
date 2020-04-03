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

# load premade 
model1 = ut.load_pr_model('model13_1')
model2 = ut.load_pr_model('model16')
model3 = ut.load_pr_model('model21')
model4 = ut.load_pr_model('model26')
model5 = ut.load_pr_model('model_28')

rf = pickle.load(open('Main model\models\Combination_mod\RF_model.sav','rb'))
rfc = pickle.load(open('Main model\models\Combination_mod\RFc.sav','rb'))
rfc_d2 = pickle.load(open('Main model\models\Combination_mod\RFc_d=2.sav','rb'))
rfc_d5 = pickle.load(open('Main model\models\Combination_mod\RFc_d=5.sav','rb'))
rfc_d3 = pickle.load(open('Main model\models\Combination_mod\RFc_d=3.sav','rb'))
rfc_d10 = pickle.load(open('Main model\models\Combination_mod\RFc_d=10.sav','rb'))
rfc_d20 = pickle.load(open('Main model\models\Combination_mod\RFc_d=20.sav','rb'))

# svc = pickle.load(open('Main model\models\Combination_mod\SVC.sav','rb'))

svc_lin = pickle.load(open('Main model\models\Combination_mod\SVC_lin.sav','rb'))
svc_lin_C0001 = pickle.load(open('Main model\models\Combination_mod\SVC_lin_C=0.001.sav','rb'))
svc_lin_C1000 = pickle.load(open('Main model\models\Combination_mod\SVC_lin_C=1000.sav','rb'))

svc100000 = pickle.load(open('Main model\models\Combination_mod\SVC_lin_iter100000.sav','rb'))
svc_rbf = pickle.load(open('Main model\models\Combination_mod\SVC_rbf.sav','rb'))
svc_poly2 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=2.sav','rb'))
svc_poly3 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=3.sav','rb'))
svc_poly5 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=5.sav','rb'))
svc_poly5_c0001 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=5_C=0001.sav','rb'))
svc_poly5_c0001_100000 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=5_C=00001_maxiter100000.sav','rb'))

train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
pre_dict = dict()
valtrain_switch = 1
if valtrain_switch ==0:
    true_labels = val_gen.classes
    val_steps = val_gen.n // val_gen.batch_size
    val_gen.reset()
    predict1 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)
    pre_dict['predict1_1'] = predict1
    predict2 = model2.predict_generator(val_gen, steps=val_steps, verbose=1)
    pre_dict['predict2'] = predict2
    predict3 = model3.predict_generator(val_gen, steps=val_steps, verbose=1)
    pre_dict['predict3'] = predict3
    predict4 = model4.predict_generator(val_gen, steps=val_steps, verbose=1)
    pre_dict['predict4'] = predict4
    predict5 = model5.predict_generator(val_gen, steps=val_steps, verbose=1)
    pre_dict['predict5'] = predict5
elif valtrain_switch==1:
    true_labels = train_gen.classes
    predict1 = np.expand_dims(np.loadtxt('Main model//Predictions//model13_1.csv', delimiter=','),axis=1)
    predict2 = np.expand_dims(np.loadtxt('Main model//Predictions//model21.csv', delimiter=','),axis=1)
    predict3 = np.expand_dims(np.loadtxt('Main model//Predictions//model16.csv', delimiter=','),axis=1)
    predict4 = np.expand_dims(np.loadtxt('Main model//Predictions//model26.csv', delimiter=','),axis=1)
    predict5 = np.expand_dims(np.loadtxt('Main model//Predictions//model28.csv', delimiter=','),axis=1)
    pre_dict['predict1_1'] = predict1
    pre_dict['predict2'] = predict2
    pre_dict['predict3'] = predict3
    pre_dict['predict4'] = predict4
    pre_dict['predict5'] = predict5

pred_matrix = np.concatenate((predict1,predict2,predict3,predict4,predict5),axis=1)
# predict5 = model4.predict_generator(val_gen, steps=val_steps,verbose=1)
# pre_dict['predict5'] = predict5

pre_dict['predict mean 5'] = np.mean([predict1,predict2,predict3,predict4,predict5],axis=0)
pre_dict['predict_min'] = pred_matrix.min(axis=1)
pre_dict['predict_max'] = pred_matrix.max(axis=1)

pre_svc_lin_C0001 = svc_lin_C0001.decision_function(pred_matrix)
pre_dict['SVC lin C=0.001'] = pre_svc_lin_C0001
pre_svc_lin_C1000 = svc_lin_C1000.decision_function(pred_matrix)
pre_dict['SVC lin C=1000'] = pre_svc_lin_C1000
# pre_svc_lin = svc_lin.decision_function(pred_matrix)
# pre_dict['SVC lin'] = pre_svc_lin
# pre_svc_100000 = svc100000.decision_function(pred_matrix)
# pre_dict['SVC iter=100000'] = pre_svc_100000
# pre_svc_rbf = svc_rbf.decision_function(pred_matrix)
# pre_dict['SVC_rbf'] = pre_svc_rbf
# pre_svc_poly2 = svc_poly2.decision_function(pred_matrix)
# pre_dict['SVC_poly2'] = pre_svc_poly2
# pre_svc_poly3 = svc_poly3.decision_function(pred_matrix)
# pre_dict['SVC_poly3'] = pre_svc_poly3
# pre_svc_poly5 = svc_poly5.decision_function(pred_matrix)
# pre_dict['SVC_poly5'] = pre_svc_poly5
# pre_svc_poly5_C0001 = svc_poly5_c0001.decision_function(pred_matrix)
# pre_dict['SVC_poly5_C=0.001'] = pre_svc_poly5_C0001
# pre_svc_poly5_C0001_100000 =  svc_poly5_c0001_100000.decision_function(pred_matrix)
# pre_dict['SVC_poly5_C=0.001,iter=100000'] = pre_svc_poly5_C0001_100000

pre_rf = rf.predict(pred_matrix)
pre_dict['RF'] = pre_rf#[:,1]
pre_rfc = rfc.predict_proba(pred_matrix)
pre_dict['RFc'] = pre_rfc[:,1]
pre_rfc_d2 = rfc_d2.predict_proba(pred_matrix)
pre_dict['RFc d=2'] = pre_rfc_d2[:,1]
pre_rfc_d3 = rfc_d3.predict_proba(pred_matrix)
pre_dict['RFc d=3'] = pre_rfc_d3[:,1]
pre_rfc_d5 = rfc_d5.predict_proba(pred_matrix)
pre_dict['RFc d=5'] = pre_rfc_d5[:,1]
pre_rfc_d10 = rfc_d10.predict_proba(pred_matrix)
pre_dict['RFc d=10'] = pre_rfc_d10[:,1]
pre_rfc_d20 = rfc_d20.predict_proba(pred_matrix)
pre_dict['RFc d=20'] = pre_rfc_d20[:,1]
# score = rf.score(pred_matrix,true_labels)

pre_dict['hyper']= np.mean([pre_rfc_d5[:,1],pre_svc_lin_C0001],axis=0)
for pr in pre_dict.keys():
    fpr, tpr, thresholds = roc_curve(true_labels, pre_dict[pr])
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
    plt.title(pr)
plt.show()