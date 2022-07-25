import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.transformers import Transformer, TransformerPositionEncoding
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

    x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float32)
    x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float32)

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

    x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float32)

    input_shape = x_train.shape[1:]
    n_classes = len(np.unique(y))

    learning_rate = 1e-4

    loss = tf.keras.losses.categorical_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model = Transformer(input_shape,
        head_size=640,
        num_heads=1,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[296, 148, 74],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=n_classes
    )


    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    earlystopping = EarlyStopping(
        monitor='val_loss', # set monitor metrics
        min_delta=0.001, # set minimum metrics delta
        patience=4, # number of epochs to stop training
        restore_best_weights=True, # set if use best weights or last weights
        )
    callbacksList = [earlystopping] # build callbacks list
    #%%
    fittedModel = model.fit(x_train, y_train, epochs=epochs, batch_size=10,
                validation_data=(x_valid, y_valid), callbacks=callbacksList) 

    probs       = model.predict(x_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_acc_loss_keras(fittedModel, y_test.argmax(axis=-1), preds, "Transformer")

    metrics_results.append({
        "model": "Transformer",
        "dataset": dataset,
        "subject": subject,
        "acc_train": fittedModel.history['accuracy'][-1],
        "acc_val": fittedModel.history['val_accuracy'][-1],
        "acc_test": acc,
    })



    model = TransformerPositionEncoding(
        input_shape=input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=n_classes
    )


    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    earlystopping = EarlyStopping(
        monitor='val_loss', # set monitor metrics
        min_delta=0.001, # set minimum metrics delta
        patience=4, # number of epochs to stop training
        restore_best_weights=True, # set if use best weights or last weights
        )
    callbacksList = [earlystopping] # build callbacks list
    #%%
    fittedModel = model.fit(x_train, y_train, epochs=epochs, batch_size=10,
                validation_data=(x_valid, y_valid), callbacks=callbacksList) 

    probs       = model.predict(x_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

    plot_acc_loss_keras(fittedModel, y_test.argmax(axis=-1), preds, "Transformer Positional")

    metrics_results.append({
        "model": "Transformer Positional",
        "dataset": dataset,
        "subject": subject,
        "acc_train": fittedModel.history['accuracy'][-1],
        "acc_val": fittedModel.history['val_accuracy'][-1],
        "acc_test": acc,
    })



if __name__ == '__main__':
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
    result.to_csv('src/artifacts/results_Transformer_models.csv', index=False, header=True)