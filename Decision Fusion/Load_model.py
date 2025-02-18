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
model_comb = ut.load_com_model('modelcombi')

rf = pickle.load(open('models\Combination_mod\RF_model_reg_restr_500_10_2.sav','rb'))
rf5 = pickle.load(open('models\Combination_mod\RF_model_reg_restr_500_10_5in.sav','rb'))

svm = pickle.load(open('models\Combination_mod\SVM.sav','rb'))
svm10 = pickle.load(open('models\Combination_mod\SVM_C=10.sav','rb'))
nusvm = pickle.load(open('models\Combination_mod\\NuSVM.sav','rb'))
nusvm100000 = pickle.load(open('models\Combination_mod\\NuSVM_100000.sav','rb'))
nusvm100000_5in = pickle.load(open('models\Combination_mod\\NuSVM_5in_100000.sav','rb'))

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

pre_dict['predict_mean'] = np.mean([predict1,predict2,predict_tran],axis=0)
pre_dict['predict_min'] = np.minimum(predict1,predict2,predict_tran)
pre_dict['predict_max'] = np.maximum(predict1,predict2,predict_tran)

pre_dict['predict_mean5'] = np.mean([predict1,predict2,predict_tran,predict4,predict5],axis=0)
# pre_dict['predict_min5'] = np.argmin([predict1,predict2,predict_tran,predict4,predict5],axis=1)
# pre_dict['predict_max5'] = np.argmax([predict1,predict2,predict_tran,predict4,predict5],axis=1)

pred_matrix = np.concatenate((predict1,predict2,predict_tran),axis=1)
pred_matrix5 = np.concatenate((predict1,predict2,predict_tran,predict4,predict5),axis=1)

print(pred_matrix.shape)
print('type predmatrix',type(pred_matrix))
pre_comb = model_comb.predict([predict1,predict2,predict_tran], steps=1, verbose=1)

pre_dict['combi'] =pre_comb
#
pre_rf = rf.predict(pred_matrix)
pre_rf5 = rf5.predict(pred_matrix5)
pre_dict['RF'] = pre_rf#[:,1]
pre_dict['RF5'] = pre_rf5


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

score = rf.score(pred_matrix,true_labels)
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