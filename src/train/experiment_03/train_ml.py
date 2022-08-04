from sklearnex import patch_sklearn
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
patch_sklearn()


from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from models.mlmodels import KNN_model_predict, LDA_model_predict, SVM_model_predict, XGB_model_predict, KNNDWN_model_predict
from utils.plots import get_metrics
from utils.data_preprocessing_utils import Utils
from utils.feature_extraction import time_domain_features, standard_scaler_transform, wavelet_transform, bandpass_transform, psd_transform



def run_ml_models(dataset, params, metrics_results):

    # ------------------------- KNN ---------------------------------------------------
    y_true, y_pred, best_score = KNN_model_predict(dataset['X_train'],
                                                   dataset['X_test'],
                                                   dataset['Y_train'],
                                                   dataset['Y_test'],
                                                   scoring=None)
    metrics_results.append(get_metrics(
        y_true, y_pred, best_score, 'KNN', params))

    # ------------------------- SVM ---------------------------------------------------
    y_true, y_pred, best_score = SVM_model_predict(dataset['X_train'],
                                                   dataset['X_test'],
                                                   dataset['Y_train'],
                                                   dataset['Y_test'],
                                                   scoring=None)
    metrics_results.append(get_metrics(
        y_true, y_pred, best_score, 'SVM', params))

    # ------------------------- XGB ---------------------------------------------------
    y_true, y_pred, best_score = XGB_model_predict(dataset['X_train'],
                                                   dataset['X_test'],
                                                   dataset['Y_train'],
                                                   dataset['Y_test'],
                                                   scoring=None)
    metrics_results.append(get_metrics(
        y_true, y_pred, best_score, 'XGB', params))


SOURCE_PATH = "C:/Users/Maods/Documents/Code-Samples/Python/MI-EEG-Dataset/dataset/processed"

# Load data
# ["FC1", "FC2"], ["FC3", "FC4"], ["FC5", "FC6"]]
# channels = Utils.combinations["e"]
channels = [["FC1", "FC2"]]

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]

x, y = Utils.load(channels, subjects, base_path=SOURCE_PATH)
y = Utils.to_categorical(y)
# x, y = x[:100] , y[:100] 

# Reshape for scaling
x_reshaped = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
# Grab a test set before SMOTE
x_train, x_test, y_train, y_test = train_test_split(x_reshaped,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.20,
                                                    random_state=42)

# Scale indipendently train/test
x_train_scaled = minmax_scale(x_train, axis=1)
x_test_scaled = minmax_scale(x_test, axis=1)


# apply smote to train data
print('classes count')
print(f'before oversampling = {y_train.sum(axis=0)}')

# smote
sm = SMOTE(random_state=42)
x_train_smote, y_train = sm.fit_resample(x_train_scaled, y_train)
print('classes count')
print(f'after oversampling = {y_train.sum(axis=0)}')


# Reshape for original format (x, 640, 2)
x_train = x_train_smote.reshape(x_train_smote.shape[0], int( x_train_smote.shape[1]/2), 2).astype(np.float32)
x_test = x_test_scaled.reshape(x_test_scaled.shape[0], int( x_test_scaled.shape[1]/2), 2).astype(np.float32)


metrics_results = []


# ===> Feture Extraction A
params = dict(
    fs=160,
    dataset='Motor Imaginary',
    subject='All subjects',
    pipeline='time domain features -> wavelet'
)
features_a_train = wavelet_transform(x_train, params)
features_a_test = wavelet_transform(x_test, params)

features_a_train = time_domain_features(features_a_train, params['fs'], ax=-2)
features_a_test = time_domain_features(features_a_test, params['fs'], ax=-2)

dataset = dict(
    X_train=features_a_train,
    X_test=features_a_test,
    Y_train=y_train,
    Y_test=y_test
)

run_ml_models(dataset, params, metrics_results)


# ===> Feture Extraction B
params = dict(
    fs=160,
    dataset='Motor Imaginary',
    subject='All subjects',
    pipeline='PCA'
)

n_components=60
pca = PCA(n_components=n_components)

features_B_train =  x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
features_B_test =  x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

features_B_train = pca.fit_transform(features_B_train)
features_B_test = pca.transform(features_B_test)

dataset = dict(
    X_train=features_B_train,
    X_test=features_B_test,
    Y_train=y_train,
    Y_test=y_test
)

run_ml_models(dataset, params, metrics_results)


result = pd.DataFrame(metrics_results)
result = result.sort_values(['cross_val_score','F1_score'], ascending=False)
result.to_csv('src/artifacts/results_ml_models.csv', index=False, header=True)

    



