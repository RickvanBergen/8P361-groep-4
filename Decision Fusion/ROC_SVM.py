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

model1 = ut.load_pr_model('model1')
model2 = ut.load_pr_model('model2')
model3 = ut.load_pr_model('transfer_model')
model4 = ut.load_pr_model('4_Model_2+conv(128)')
model5 = ut.load_pr_model('13_Model_12+lr(0.001)')
model31 = ut.load_pr_model('Model31')

train_gen, val_gen = get_pcam_generators(
    r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\\")
true_labels = val_gen.classes
val_steps = val_gen.n//val_gen.batch_size
val_gen.reset()

pre_dict = dict()
predict1 = model1.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict1'] = predict1
predict2 = model2.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict2'] = predict2
predict_tran = model3.predict_generator(val_gen, steps=val_steps, verbose=1)
pre_dict['predict_transfer'] = predict_tran
predict4 = model4.predict_generator(val_gen, steps=val_steps,verbose=1)
pre_dict['predict4'] = predict4
predict5 = model4.predict_generator(val_gen, steps=val_steps,verbose=1)
pre_dict['predict5'] = predict5
predict31 = model31.predict_generator(val_gen, steps=val_steps,verbose=1)
pre_dict['predict98'] = predict31

pre_dict['predict_mean'] = np.mean([predict1,predict2,predict_tran],axis=0)

pre_dict['predict_mean5'] = np.mean([predict1,predict2,predict_tran,predict4,predict5],axis=0)

pred_matrix = np.concatenate((predict1,predict2,predict_tran),axis=1)
pred_matrix5 = np.concatenate((predict1,predict2,predict_tran,predict4,predict5),axis=1)
pred_matrix98 = np.concatenate((predict1,predict4,predict31),axis=1)


svm = pickle.load(open('models\Combination_mod\SVM.sav','rb'))
svm10 = pickle.load(open('models\Combination_mod\SVM_C=10.sav','rb'))
nusvm = pickle.load(open('models\Combination_mod\\NuSVM.sav','rb'))
nusvm100000 = pickle.load(open('models\Combination_mod\\NuSVM_100000.sav','rb'))
nusvm100000_5in = pickle.load(open('models\Combination_mod\\NuSVM_5in_100000.sav','rb'))
nusvm_5in = pickle.load(open('models\Combination_mod\\NuSVM_5in.sav','rb'))
nusvm_rbf = pickle.load(open('models\Combination_mod\\NuSVM_rbf.sav','rb'))
nusvm_rbf_98 = pickle.load(open('models\Combination_mod\\NuSVM_98_rbf.sav','rb'))
nusvm_98 = pickle.load(open('models\Combination_mod\\NuSVM_98.sav','rb'))
nusvm_98_01 = pickle.load(open('models\Combination_mod\\NuSVM_98_nu=0,1.sav','rb'))
nusvm_98_05 = pickle.load(open('models\Combination_mod\\NuSVM_98_nu=0,5.sav','rb'))
nusvm_98_07 = pickle.load(open('models\Combination_mod\\NuSVM_98_nu=0,7.sav','rb'))
nusvm_98_1 = pickle.load(open('models\Combination_mod\\NuSVM_98_nu=1.sav','rb'))
nusvm_98_1 = pickle.load(open('models\Combination_mod\\NuSVM_98_nu=1.sav','rb'))
svc_5in = pickle.load(open('models\Combination_mod\SVC_pr5.sav','rb'))
svc_5in_rbf = pickle.load(open('models\Combination_mod\SVC_pr5_rbf.sav','rb'))
svc_5in_C100 = pickle.load(open('models\Combination_mod\SVC_pr5_C=100.sav','rb'))
svc_5in_iter_10000 = pickle.load(open('models\Combination_mod\SVC_pr5_iter=10000.sav','rb'))
svc_poly_2 = pickle.load(open('models\Combination_mod\SVC_poly2.sav','rb'))
svc_poly_3 = pickle.load(open('models\Combination_mod\SVC_poly3.sav','rb'))
svc_poly_4 = pickle.load(open('models\Combination_mod\SVC_poly4.sav','rb'))
svc_poly_5 = pickle.load(open('models\Combination_mod\SVC_poly5.sav','rb'))


pre_svm = svm.predict(pred_matrix)
pre_dict['SVM'] = pre_svm
pre_svm10 = svm.predict(pred_matrix)
pre_dict['SVM 10'] = pre_svm10
pre_nusvm = nusvm.predict(pred_matrix)
pre_dict['NuSVM'] = pre_nusvm
pre_nusvm100000 = nusvm100000.predict(pred_matrix)
pre_dict['NuSVM_100000'] = pre_nusvm100000
pre_nusvm100000_5in = nusvm100000_5in.predict(pred_matrix5)
pre_dict['NuSVM_100000_5in'] = pre_nusvm100000_5in
pre_nusvm_5in = nusvm_5in.predict(pred_matrix5)
pre_dict['NuSVM_5in'] = pre_nusvm_5in
pre_nusvm_rbf = nusvm_rbf.predict(pred_matrix)
pre_dict['NuSVM_rbf'] = pre_nusvm_rbf
pre_nusvm_rbf_98 = nusvm_rbf_98.predict(pred_matrix98)
# pre_dict['NuSVM_rbf_98'] = pre_nusvm_rbf_98
# pre_nusvm_98 = nusvm_98.predict(pred_matrix98)
# pre_dict['NuSVM_98'] = pre_nusvm_98
# pre_nusvm_98_01 = nusvm_98_01.predict(pred_matrix98)
# pre_dict['NuSVM_98_nu=0.1'] = pre_nusvm_98_01
# pre_nusvm_98_05 = nusvm_98_05.predict(pred_matrix98)
# pre_dict['NuSVM_98_nu=0.5'] = pre_nusvm_98_05
# pre_nusvm_98_07 = nusvm_98_07.predict(pred_matrix98)
# pre_dict['NuSVM_98_nu=0.7'] = pre_nusvm_98_07
# # pre_nusvm_98_1 = nusvm_98_1.predict(pred_matrix98)
# pre_dict['NuSVM_98_nu=1'] = pre_nusvm_98_1
pre_svc_5in = svc_5in.decision_function(pred_matrix5)
pre_dict['SVC_5in'] = pre_svc_5in
pre_svc_5in_rbf = svc_5in_rbf.decision_function(pred_matrix5)
pre_dict['SVC_5in_rbf'] = pre_svc_5in_rbf
pre_svc_5in_C100 = svc_5in_C100.decision_function(pred_matrix5)
pre_dict['SVC_5in_C100'] = pre_svc_5in_C100
pre_svc_5in = svc_5in.decision_function(pred_matrix5)
pre_dict['SVC_5in'] = pre_svc_5in
pre_svc_poly2 = svc_poly_2.decision_function(pred_matrix)
pre_dict['SVC poly 2'] = pre_svc_poly2
pre_svc_poly3 = svc_poly_3.decision_function(pred_matrix)
pre_dict['SVC poly 3'] = pre_svc_poly3
pre_svc_poly4 = svc_poly_4.decision_function(pred_matrix)
pre_dict['SVC poly 4'] = pre_svc_poly4
pre_svc_poly5 = svc_poly_5.decision_function(pred_matrix)
pre_dict['SVC poly 5'] = pre_svc_poly5


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