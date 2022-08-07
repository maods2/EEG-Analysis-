# Data Science With EEG Signals
This repository is part of the conclusion work underdevelopment on the Data Science & Analytics post-graduate degree by Centro Universitário SENAI CIMATEC.
The study consists of classifying EEG signals Motor Imaginary, intending to explore and compare the Transformers model-based architectures with other traditional Machine Learning algorithms.

A brief overview of the file structure for the repository:

```bash

│   .gitignore   
│   LICENSE
│   Makefile   # Make file with some scripts for automation
│   README.md
│   requirements.txt
│
├───scripts # scripts to automate the models running process 
│       run_all.sh
│       run_cnns.sh
│       run_ml.sh
│       run_transformers.sh
│
└───src
    │   __init__.py
    │
    ├───artifacts # Artifacts genereted after training, such as figures and csvs 
    │
    ├───models
    │       cnn_models.py # CNNs Architectures
    │       mlmodels.py # ML Pipelines
    │       transformers.py # Transformers Architectures
    │       __init__.py
    │
    ├───notebooks # Python Notebooks with Exploratory data analysis and plot results from the models
    │
    ├───train # Store the train files for each models 
    │   │   train_cnn.py
    │   │   train_ml.py
    │   │   train_transformer.py
    │   │   __init__.py
    │   ├───experiment_01 # In each folder is present the training files used for the respectives experiments
    │   ├───experiment_02
    │   ├───experiment_03
    │   ├───experiment_04
    │   ├───experiment_05
    │   └───experiment_06
    │
    └───utils
            dataset_generator.py # Functions responsible download and preprocessing the dataset
            data_preprocessing_utils.py # Functions responsible for load the datasets
            feature_extraction.py  # Feature extractions methods
            plots.py # We can find here some utilities functions for plotting
            __init__.py
```


# EEG-Analysis

Despite being a poorly known technology, the Brain-Computer Interface research field has been increasing in the past few years, guided by a variety of potential solutions present in many areas. The brain-computer interface is a technique that uses electrical signals from the brain to activate external devices, such as robotics, mechanicals, or virtual artifacts, allowing users to communicate with the outside world through their minds. Brain-Computer Interface has contributed to various fields of research like Medicine, Neuroergonomics, Intelligent Environments, Neuromarketing, Advertising, Education, Self-regulation, Games, Entertainment, and  Security. Nonetheless, there are still many challenges for which the science of Brain-computer interfaces needs to transcend, such as the processes of pattern recognition and transfer learning from subject to subject when we are exploring the Artificial Intelligence area responsible for understanding the brain activity.

However, to dig into the studies of Brain-Computer Interface capabilities, it was decided for this project, first, to promote research aiming to compare different Machine Learning algorithms, from the traditional ones to the state of the art approaches, such as k-Nearest Neighbor, Support Vector Machines, Linear Discriminant Analysis, XGBoost, Convolutional Neural Networks, and Transformers models based.

<i>Before going into the code, we had to think about how we should structure the methodology for this study. That being said, It was designed the following guideline:</i>


1. Find datasets
2. Preprocess the dataset
3. Perform Exploratory Data Analysis in the datasets
4. Train Machine Learning Algorithms to have a benchmark.

    - Develop feature extraction methods
    - Find the best feature extraction pipeline for the given datasets
    - Train definitive models
    - Present results:
        >[Inter-subjct | Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/results_ml_models.csv)
    
        >[Intra-subjct | Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/results_ml_models.csv)
    

5. Train Convolutional Neural Networks to have a benchmark.

    - Implement 1d and 2d CNNs
    - Train definitive models
    - Present results: 
        >[Inter-subjct | Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/results_cnn_models.csv)
        
        >[Intra-subjct | Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/results_cnn_models.csv)


6. Traine Transformer-based models.
    - Implement Architecture
    - Test new approaches
    - Present results: 
        >[Inter-subjct | Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/results_Transformer_models.csv)
      
        >[Intra-subjct | Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/results_Transformer_models.csv)
    

## 1. Finding datasets

The datasets were found in a GitHub repository that lists different sources of EEG data and their different areas of applicability. 

 [A list of all public EEG-datasets](https://github.com/meagmohit/EEG-Datasets)

Firstly, the plan was to choose only two different datasets but after some tests with the feature extraction process along with the traditional Machine Learning algorithms, we noticed poor performance, even after a huge effort in the data preprocessing. Therefore, in order to concentrate effort only in one problem we took the decision of being working only with Motor Movement / Imagery dataset.

The following data set was selected:



    - EEG Motor Movement/Imagery Dataset (Sept. 9, 2009)
        64 EEG channels  
        4 classes, 80 subjects
        160Hz sampling rate

    
                B indicates baseline
                L indicates motor imagination of opening and closing left fist;
                R indicates motor imagination of opening and closing right fist;
                LR indicates motor imagination of opening and closing both fists;
                F indicates motor imagination of opening and closing both feet.


**Dataset documentation:**

- [EEG Motor Movement/Imagery Dataset (Sept. 9, 2009)](https://www.physionet.org/content/eegmmidb/1.0.0/)


## 2. Preprocessing the dataset
Once we already had the data, the next step would be to understand its structure in order to process it accordingly to their needs. The datasets is available in EDF format (The European Data Format), which is a simple and flexible format for the exchange and storage of multichannel biological and physical signals. In order to work with this data formats, the MNE library was used. MNE provides a toolkit for loading and processing biological signals.

Although we have functions that load the type of data mentioned above, it was still necessary to develop some functions in order to reshape and aggregate the signal in the following format. During the project we found out a github repository that implemented various functions that are used to load that specifc dataset. Taking this into account, we used those functions in our project ([MI-EEG-1D-CNN](https://github.com/Kubasinska/MI-EEG-1D-CNN)).
After preprocessing, the data was provided in the following format for the models:

     (trails, channels, samples)

## 3. Performing Exploratory Data Analysis in the datasets
Since we already have the proper ways to load the data, the next step that we should go through is data analysis, which for the present scenario, we may consider to be a time series analysis. The proposed strategy is to perform the analysis on three different levels, comparing "datasets", "classes" and "channels".

Level of comparison:

    > Dataset
        > Classes
            > Channels

At the beginning of this project, we created some sort of plots ( time-series, Fourier, Spectrum) for this specific section but currently, we are going to restructure the analysis based on the following methods:

(Data Normalized)
- Autocorrelation
- Stationarity test
- Time-domain analysis -> compute statistics / Distributions
- Frequency-domain analysis -> compute statistics  / Distributions
    - FFT
- Frequency/time-domain analysis ->
    - Wavelet 
    - STFT
- Cluster analysis + Multivariate Analysis (PCA) -> Plot clusters 

## 4. Train Machine Learning Algorithms to have a benchmark.
As long as we test state-of-the-art algorithms we need to have some sort of metrics baseline considering the traditional literature models. In order to accomplish that, before start playing with Deep Learning, we will explored some Traditional Machine Learning algorithms. The idea here is to implement lots of feature extraction methods and try out various combinations of pipelines to understand the better approach for the given datasets. Each dataset will be trained using two strategies, firstly trying only one individual per training session, and secondly passing data from all individuals at once.
In addition, we will choose the more performative pipeline to define as our benchmark. After testing more than 20 different pipelines, we selected the following to proceed with the experiments:

    > Traditional Machine Learning Pipelines
        - PCA + KNN
        - PCA + XGBoost
        - PCA + SVM
        - Wavelet Decomposition + Time Features + KNN
        - Wavelet Decomposition + Time Features + XGBoost
        - Wavelet Decomposition + Time Features + SVM


### ML Results | Inter-Subject Approach 
![Inter-subject ML results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/barplot_ml.png "Title")
<br></br>
### ML Results | Intra-Subject Approach 
![Inter-subject ML results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/barplot_ml.png "Title")

## 5. Train Convolutional Neural Networks to have a benchmark.
Similar to the previous section, now we will implement three different Convolution Neural Networks and test the data, using the same strategy of training only with one individual data, then training with the whole dataset. The CNNs were selected on the following GitHub repositories.

[Army Research Laboratory (ARL) EEGModels project: A Collection of Convolutional Neural Network (CNN) models for EEG signal processing and classification](https://github.com/vlawhern/arl-eegmodels) / [MI-EEG-1D-CNN](https://github.com/Kubasinska/MI-EEG-1D-CNN)

    > CNNs
        - EEGNet
        - DeepConvNet
        - ShallowConvNet
        - HopefullNet

### CNNs Results | Inter-Subject Approach 
![Inter-subject CNN results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/barplot_cnn.png "Title")
<br></br>
### CNNs Results | Intra-Subject Approach 
![Inter-subject CNN results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/barplot_cnn.png "Title")


## 6. Traine Transformer-based models
In this section of the study, we implemented two types of Transformers archtectures for the EEG classification. That being said, for this specific application we are only using the Transformer Encoder coupled with a fully connected layer and sofmax function. One of the architectures was implemented without positional encoder and the second one with it 

    > Transformers based models
        - Transformer
        - Transformer + Positional Encoder
    

### Transformers Results | Inter-Subject Approach 
![Inter-subject Transformers results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/barplot_trans.png "Title")
<br></br>
### Transformers Results | Intra-Subject Approach 
![Inter-subject Transformers results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/barplot_trans.png "Title")


## 7. Comparison of all models

### All Models Results | Inter-Subject Approach 
![Inter-subject All Models results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_05/barplot_all.png "Title")
<br></br>
### All Models Results | Intra-Subject Approach 
![Inter-subject All Models results](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/experiment_06/barplot_all.png "Title")


<br><br>
## Code Usage - Creating Virtualenv

Before you begin, ensure you have met the following requirements (Use the guide below to install virtual env with the required libraries.):
<!--- These are just example requirements. Add, duplicate or remove as required --->
* Python version: `Python==3.9.12`
* [requirements.txt](https://github.com/maods2/EEG-Analysis-/blob/main/requirements.txt)

> Instaling
>  -
```cmd
python -m pip install --user virtualenv
```
> Creating new env
>  -
```cmd
python -m venv env_tcc_eeg

```
> Activating env
>  -
```cmd
.\env_tcc_eeg\Scripts\activate

```
> Leaving virtual env
>  -
```cmd
deactivate
```

> Updating pip
>  -
```cmd
python -m pip install --upgrade pip
```

> Installing packages
>  -
```cmd
python -m pip install -r requirements.txt


```
[Virtual environments reference](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
