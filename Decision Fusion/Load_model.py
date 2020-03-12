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
                                            class_mode='binary')

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
pre_dict['predict_mean'] = np.mean([predict1,predict2,predict_tran],axis=0)
pre_dict['predict_min'] = np.minimum(predict1,predict2,predict_tran)
pre_dict['predict_max'] = np.maximum(predict1,predict2,predict_tran)

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