'''
TU/e BME Project Imaging 2019
Submission code for Kaggle PCAM
Author: Suzanne Wetstein
'''

import os

import numpy as np

import glob
import pandas as pd
from matplotlib.pyplot import imread
import os
from keras.models import model_from_json
import pickle

#Load test data filepath
TEST_PATH = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\test\test"
#Load in individual model filepaths
MODEL_FILEPATH1 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\Model13_1.json"
MODEL_FILEPATH2 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\model16.json"
MODEL_FILEPATH3 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\Model26.json"
MODEL_FILEPATH4 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\model_28.json"
MODEL_FILEPATH5 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\model21.json"

#Load in ensemble model filepaths
combi_model_filepath = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Combination_mod\RFc_d=5.sav"
combi_model_filepath_2 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Combination_mod\SVC_lin_C=0.001.sav"

#Load in filepaths of weights for the individual models
MODEL_WEIGHTS_FILEPATH1 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\Model13_1_weights.hdf5"
MODEL_WEIGHTS_FILEPATH2 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\model16_weights.hdf5"
MODEL_WEIGHTS_FILEPATH3 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\Model26_weights.hdf5"
MODEL_WEIGHTS_FILEPATH4 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\model_28_weights.hdf5"
MODEL_WEIGHTS_FILEPATH5 = r"C:\Users\20174099\Documents\School\Jaar 3\Imaging Project\8P361-groep-4\Main model\models\Prediction_mod\model21_weights.hdf5"



# load model and model weights

#json_file = open(MODEL_FILEPATH, 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
with open(MODEL_FILEPATH1, 'r') as f:
    model1 = model_from_json(f.read())
with open(MODEL_FILEPATH2, 'r') as f:
    model2 = model_from_json(f.read())
with open(MODEL_FILEPATH3, 'r') as f:
    model3 = model_from_json(f.read())
with open(MODEL_FILEPATH4, 'r') as f:
    model4 = model_from_json(f.read())
with open(MODEL_FILEPATH5, 'r') as f:
    model5 = model_from_json(f.read())
    
# load weights into new model
model1.load_weights(MODEL_WEIGHTS_FILEPATH1)
model2.load_weights(MODEL_WEIGHTS_FILEPATH2)
model3.load_weights(MODEL_WEIGHTS_FILEPATH3)
model4.load_weights(MODEL_WEIGHTS_FILEPATH4)
model5.load_weights(MODEL_WEIGHTS_FILEPATH5)


#Load RandomForest and or SVC
rfc = pickle.load(open(combi_model_filepath,'rb'))
svc = pickle.load(open(combi_model_filepath_2,'rb'))
# open the test set in batches (as it is a very big dataset) and make predictions
test_files = glob.glob(os.path.join(TEST_PATH, '*.tif'))
submission = pd.DataFrame()
file_batch = 5000
print(len(test_files))
max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):

    print('Indexes: %i - %i'%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})


    # get the image id 
    test_df['id'] = test_df.path.map(lambda x: x.split(os.sep)[-1].split('.')[0])
    test_df['image'] = test_df['path'].map(imread)
    
    
    K_test = np.stack(test_df['image'].values)
    
    # apply the same preprocessing as during draining
    K_test = K_test.astype('float')/255.0
    
    prediction1 = model1.predict(K_test)
    prediction2 = model2.predict(K_test)
    prediction3 = model3.predict(K_test)
    prediction4 = model4.predict(K_test)
    prediction5 = model5.predict(K_test)
    pred_matrix = np.concatenate((prediction1,prediction2,prediction3,prediction4,prediction5),axis=1)
    # predictions = rfc.predict_proba(pred_matrix)[:,1]
    # predictions = svc.decision_function(pred_matrix)
    rfc_pr = rfc.predict_proba(pred_matrix)[:,1]
    svc_pr = svc.decision_function(pred_matrix)
    predictions = np.mean([rfc_pr, svc_pr],axis=0)
    # predictions = pred_matrix.max(axis=1)
    # predictions = pred_matrix.max(axis=1)

    test_df['label'] = predictions
    submission = pd.concat([submission, test_df[['id', 'label']]])

# save your submission
submission.head()
submission.to_csv('submission_mean_best_rfc_svc.csv', index = False, header = True)
