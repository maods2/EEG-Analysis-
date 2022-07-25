import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.cnn_models import EEGNet, ShallowConvNet, DeepConvNet, HopefullNet
from utils.plots import plot_acc_loss_keras
from utils.dataloader import load_eeg_data_edf, load_eeg_data_gdf, load_eeg_data_mat
from utils.dataloader import load_eeg_data_mat
from sklearn.preprocessing import minmax_scale
from imblearn.over_sampling import SMOTE
from utils.data_preprocessing_utils import Utils
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import  EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import tensorflow as tf



def run_pipeline(nb_classes, chans, samples, dataset, subject, metrics_results, kernels=1, epochs=50):
    
    # take 50/25/25 percent of the data to train/validate/test
    SOURCE_PATH = "C:/Users/Maods/Documents/Code-Samples/Python/MI-EEG-Dataset/dataset/processed"

    # Load data
    # ["FC1", "FC2"], ["FC3", "FC4"], ["FC5", "FC6"]]
    channels = Utils.combinations["e"]

    exclude = [38, 88, 89, 92, 100, 104]
    subjects = [n for n in np.arange(1, 110) if n not in exclude]

    x, y = Utils.load(channels, subjects, base_path=SOURCE_PATH)
    # x, y = x[:500] , y[:500] 
    #Transform y to one-hot-encoding
    y_one_hot  = Utils.to_one_hot(y, by_sub=False)
    #Reshape for scaling
    reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    #Grab a test set before SMOTE
    x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                                y_one_hot,
                                                                                stratify=y_one_hot,
                                                                                test_size=0.20,
                                                                                random_state=42)

    #Scale indipendently train/test
    x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
    x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

    #Create Validation/test
    x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                        y_valid_test_raw,
                                                        stratify=y_valid_test_raw,
                                                        test_size=0.50,
                                                        random_state=42)

    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0],2, int(x_valid_raw.shape[1]/2)).astype(np.float32)
    x_test = x_test_raw.reshape(x_test_raw.shape[0],2, int(x_test_raw.shape[1]/2)).astype(np.float32)

    #apply smote to train data
    print('classes count')
    print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
    # smote
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
    print('classes count')
    print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
    print ('after oversampling = {}'.format(y_train.sum(axis=0)))

    x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], 2, int(x_train_smote_raw.shape[1]/2)).astype(np.float32)


        

    if nb_classes == 2:
         class_weights = {0:1, 1:1}
    if nb_classes == 3:
         class_weights = {0:1, 1:1, 2:1}
    if nb_classes == 4:
         class_weights = {0:1, 1:1, 2:1, 3:1}
    if nb_classes == 5:
         class_weights = {0:1, 1:1, 2:1, 3:1, 4:1}

    #-------------------------   EEGNet ---------------------------------------------------

    model = EEGNet(nb_classes = nb_classes, Chans = chans, Samples = samples, 
                dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])



    

    fittedModel = model.fit(x_train, y_train, batch_size = 16, epochs = epochs, 
                            verbose = 2, validation_data=(x_valid, y_valid), class_weight = class_weights)

    probs       = model.predict(x_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_acc_loss_keras(fittedModel, y_test.argmax(axis=-1), preds, "EEGNet")

    metrics_results.append({
        "model": "EEGNet",
        "dataset": dataset,
        "subject": subject,
        "acc_train": fittedModel.history['accuracy'][-1],
        "acc_val": fittedModel.history['val_accuracy'][-1],
        "acc_test": acc,
    })

    #-------------------------   ShallowConvNet ---------------------------------------------------
    model = ShallowConvNet(nb_classes = nb_classes, Chans = chans, Samples = samples)

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])


   

    fittedModel = model.fit(x_train, y_train, batch_size = 16, epochs = epochs, 
                            verbose = 2, validation_data=(x_valid, y_valid), class_weight = class_weights)

    probs       = model.predict(x_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_acc_loss_keras(fittedModel, y_test.argmax(axis=-1), preds, "ShallowConvNet")

    metrics_results.append({
        "model": "ShallowConvNet",
        "dataset": dataset,
        "subject": subject,
        "acc_train": fittedModel.history['accuracy'][-1],
        "acc_val": fittedModel.history['val_accuracy'][-1],
        "acc_test": acc,
    })

    #-------------------------   DeepConvNet ---------------------------------------------------


    model = DeepConvNet(nb_classes = nb_classes, Chans = chans, Samples = samples)

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])


    fittedModel = model.fit(x_train, y_train, batch_size = 16, epochs = epochs, 
                            verbose = 2, validation_data=(x_valid, y_valid), class_weight = class_weights)

    probs       = model.predict(x_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_acc_loss_keras(fittedModel, y_test.argmax(axis=-1), preds, "DeepConvNet")

    metrics_results.append({
        "model": "DeepConvNet",
        "dataset": dataset,
        "subject": subject,
        "acc_train": fittedModel.history['accuracy'][-1],
        "acc_val": fittedModel.history['val_accuracy'][-1],
        "acc_test": acc,
    })
    #-------------------------   HopefullNet ---------------------------------------------------
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[-1],x_train.shape[-2]).astype(np.float32)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[-1],x_valid.shape[-2]).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[-1],x_test.shape[-2]).astype(np.float32)

    learning_rate = 1e-4

    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model =  HopefullNet()

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    earlystopping = EarlyStopping(
            monitor='val_loss', # set monitor metrics
            min_delta=0.001, # set minimum metrics delta
            patience=4, # number of epochs to stop training
            restore_best_weights=True, # set if use best weights or last weights
            )
    callbacksList = [ earlystopping] 
    

    fittedModel = model.fit(x_train, y_train, epochs=epochs, batch_size=10,
                validation_data=(x_valid, y_valid), callbacks=callbacksList) 

    probs       = model.predict(x_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_acc_loss_keras(fittedModel, y_test.argmax(axis=-1), preds, "HopefullNet")

    metrics_results.append({
        "model": "HopefullNet",
        "dataset": dataset,
        "subject": subject,
        "acc_train": fittedModel.history['accuracy'][-1],
        "acc_val": fittedModel.history['val_accuracy'][-1],
        "acc_test": acc,
    })


metrics_results = []

run_pipeline(
    nb_classes=5, 
    chans=2, 
    samples=640, 
    dataset="Motor Imaginary", 
    subject="All",
    metrics_results=metrics_results, 
    kernels=1, 
    epochs=50
    )


result = pd.DataFrame(metrics_results)
result = result.sort_values(['acc_test','acc_val'], ascending=False)
print(result.head(30))
result.to_csv('src/artifacts/results_cnn_models.csv', index=False, header=True)