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

model1 = ut.load_pr_model('model13')
model1_1 = ut.load_pr_model('model13_1')
model2 = ut.load_pr_model('model31')
model3 = ut.load_pr_model('model26')
model4 = ut.load_pr_model('model16')
model5 = ut.load_pr_model('transfer_MobileNetV2')
model6 = ut.load_pr_model('model_28')
model7 = ut.load_pr_model('model_30')
# model4 = ut.load_pr_model('4_Model_2+conv(128)')
# model5 = ut.load_pr_model('13_Model_12+lr(0.001)')
# model_comb = ut.load_com_model('modelcombi')

# rf = pickle.load(open('models\Combination_mod\RF_model_reg_restr_500_10_2.sav','rb'))
# rf5 = pickle.load(open('models\Combination_mod\RF_model_reg_restr_500_10_5in.sav','rb'))

svc = pickle.load(open('Main model\models\Combination_mod\SVC.sav','rb'))
svc100000 = pickle.load(open('Main model\models\Combination_mod\SVC_iter=100000.sav','rb'))
svc_rbf = pickle.load(open('Main model\models\Combination_mod\SVC_rbf.sav','rb'))
svc_rbf_C100 = pickle.load(open('Main model\models\Combination_mod\SVC_rbf_C=100.sav','rb'))
svc_rbf_C1000 = pickle.load(open('Main model\models\Combination_mod\SVC_rbf_C=1000.sav','rb'))
svc_rbf_C01 = pickle.load(open('Main model\models\Combination_mod\SVC_rbf_C=0.1.sav','rb'))
svc_rbf_C001 = pickle.load(open('Main model\models\Combination_mod\SVC_rbf_C=0.01.sav','rb'))
svc_rbf_C00001 = pickle.load(open('Main model\models\Combination_mod\SVC_rbf_C=0.01.sav','rb'))
svc_poly2 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=2.sav','rb'))
svc_poly3 = pickle.load(open('Main model\models\Combination_mod\SVC_poly=3.sav','rb'))

# svm10 = pickle.load(open('models\Combination_mod\SVM_C=10.sav','rb'))
# nusvm = pickle.load(open('models\Combination_mod\\NuSVM.sav','rb'))
# nusvm100000 = pickle.load(open('models\Combination_mod\\NuSVM_100000.sav','rb'))
# nusvm100000_5in = pickle.load(open('models\Combination_mod\\NuSVM_5in_100000.sav','rb'))

train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
true_labels = val_gen.classes
val_steps = val_gen.n//val_gen.batch_size
val_gen.reset()
pre_dict = dict()
predict1 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict1'] = predict1
predict1_1 = model1_1.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict1_1'] = predict1_1
predict2 = model2.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict2'] = predict2
predict3 = model3.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict3'] = predict3
predict4 = model4.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict4'] = predict4
predict5 = model5.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['transfer MobileNetV2'] = predict5
predict6 = model6.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict6'] = predict6
predict7 = model7.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict7'] = predict7
# predict4 = model4.predict_generator(val_gen, steps=val_steps,verbose=1)
# pre_dict['predict4'] = predict4

# predict5 = model4.predict_generator(val_gen, steps=val_steps,verbose=1)
# pre_dict['predict5'] = predict5

pre_dict['predict_mean'] = np.mean([predict1,predict2,predict3],axis=0)
pre_dict['predict mean 5'] = np.mean([predict1,predict2,predict3,predict4,predict5],axis=0)
pre_dict['predict mean 8'] = np.mean([predict1,predict1_1,predict2,predict3,predict4,predict5,predict6,predict7],axis=0)
pre_dict['predict_min'] = np.minimum(predict1,predict2,predict3)
pre_dict['predict_max'] = np.maximum(predict1,predict2,predict3)
#
# pre_dict['predict_mean5'] = np.mean([predict1,predict2,predict_tran,predict4,predict5],axis=0)
# pre_dict['predict_min5'] = np.argmin([predict1,predict2,predict_tran,predict4,predict5],axis=1)
# pre_dict['predict_max5'] = np.argmax([predict1,predict2,predict_tran,predict4,predict5],axis=1)

pred_matrix = np.concatenate((predict1,predict2,predict3),axis=1)
# pred_matrix5 = np.concatenate((predict1,predict2,predict_tran,predict4,predict5),axis=1)

# print(pred_matrix.shape)
# print('type predmatrix',type(pred_matrix))
# pre_comb = model_comb.predict([predict1,predict2,predict_tran], steps=1, verbose=1)
#
# pre_dict['combi'] =pre_comb
#
# pre_rf = rf.predict(pred_matrix)
# pre_rf5 = rf5.predict(pred_matrix5)
# pre_dict['RF'] = pre_rf#[:,1]
# pre_dict['RF5'] = pre_rf5


# pre_svc = svc.decision_function(pred_matrix)
# pre_dict['SVC'] = pre_svc
# pre_svc_100000 = svc100000.decision_function(pred_matrix)
# pre_dict['SVC iter=100000'] = pre_svc_100000
# pre_svc_rbf = svc_rbf.decision_function(pred_matrix)
# pre_dict['SVC_rbf'] = pre_svc_rbf
# pre_svc_rbf_C100 = svc_rbf_C100.decision_function(pred_matrix)
# pre_dict['SVC_rbf_ C=100'] = pre_svc_rbf_C100
# pre_svc_rbf_C1000 = svc_rbf_C1000.decision_function(pred_matrix)
# pre_dict['SVC_rbf_ C=1000'] = pre_svc_rbf_C1000
# pre_svc_rbf_C01 = svc_rbf_C01.decision_function(pred_matrix)
# pre_dict['SVC_rbf_ C=0.1'] = pre_svc_rbf_C01
# pre_svc_rbf_C001 = svc_rbf_C001.decision_function(pred_matrix)
# pre_dict['SVC_rbf_ C=0.01'] = pre_svc_rbf_C001
# pre_svc_rbf_C00001 = svc_rbf_C00001.decision_function(pred_matrix)
# pre_dict['SVC_rbf_ C=0.0001'] = pre_svc_rbf_C00001
# pre_svc_poly2 = svc_poly2.decision_function(pred_matrix)
# pre_dict['SVC_poly2'] = pre_svc_poly2
# pre_svc_poly3 = svc_poly3.decision_function(pred_matrix)
# pre_dict['SVC_poly3'] = pre_svc_poly3

# score = rf.score(pred_matrix,true_labels)
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