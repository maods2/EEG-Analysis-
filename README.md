# Data Science With EEG Signals
This repository is part of the conclusion work underdevelopment on the Data Science & Analytics post-graduate degree by Centro Universitário SENAI CIMATEC.
The study consists of classifying EEG signals Motor Imaginary, intending to explore and compare the Transformers model-based architectures with other traditional Machine Learning algorithms.

A brief overview of the file structure for the repository:

```bash
|   .gitignore   
|   LICENSE
|   Makefile   # Make file with some scripts for automation
|   README.md
|   requirements.txt
|       
\---src
    |   cnn_train.ipynb   # File where we are training CNN models to get the metrics
    |   exploratory_data_analysis.ipynb # Exploratory Data Analysis for the datasets
    |   ml_pipeline_search.ipynb # Search the best feature extraction pipeline to train the benchmark 
    |   ml_train.ipynb # File where we are training ML models to get the metrics
    |   results # Plot of the results for the different models
    |   
    +---artifacts
    |       ml_models_results.csv # models serch results
    |       
    +---models
    |       cnn_models.py # CNNs Architectures
    |       transformers.py # Transformers Architectures
    |       mlmodels.py # ML Gridsearch
    |       __init__.py
    |       
    \---utils
            dataloader.py # Functions responsible for load the datasets
            feature_extraction.py # Feature extractions methods
            mlpipelinebuilder.py # Class used to organize and train different comination de models and pipelines
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
    - Present results - [Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/ml_models_results.csv)
    <br><br>

5. Train Convolutional Neural Networks to have a benchmark.

    - Implement 1d and 2d CNNs
    - Train definitive models
    - Present results [Check out the results we have achieved so far.](https://github.com/maods2/EEG-Analysis-/blob/main/src/artifacts/cnn_models_results.csv)
    <br><br>

6. Traine Transformer-based models.
    - Implement Architecture
    - Test new approaches
    - Compare results with other models
    <br><br>

## 1. Finding datasets

The datasets were found in a GitHub repository that lists different sources of EEG data and their different areas of applicability. 

 [A list of all public EEG-datasets](https://github.com/meagmohit/EEG-Datasets)

Firstly the plan was to choose only two different datasets but after some tests with the feature extraction process alongside the traditional Machine Learning algorithms showed poor performance, even after a huge effort in the data preprocessing. Therefore, in addition, one more dataset was selected aiming for better metrics as a result.

The following data sets were selected:



    - BCI Competition 2008 – Graz data set I - 
        64 EEG channels        
        2 classes (+ idle state), 7 subjects
        100Hz sampling rate

    - BCI Competition 2008 – Graz data set II A
        22 EEG channels        
        4 classes, 9 subjects
        250Hz sampling rate
    
    - EEG Motor Movement/Imagery Dataset (Sept. 9, 2009)
        64 EEG channels  
        4 classes, 80 subjects
        160Hz sampling rate



**Dataset documentation:**
- [BCI Competition 2008 – Graz data set I](https://www.bbci.de/competition/iv/desc_1.html)
- [BCI Competition 2008 – Graz data set II A](https://www.bbci.de/competition/iv/desc_2a.pdf)
- [EEG Motor Movement/Imagery Dataset (Sept. 9, 2009)](https://www.physionet.org/content/eegmmidb/1.0.0/)


## 2. Preprocessing the dataset
Once we already had the data, the next step would be to understand its structure in order to process it accordingly to their needs. The datasets are available in three different formats. The first one is EDF (The European Data Format), which is a simple and flexible format for the exchange and storage of multichannel biological and physical signals. The second one is GDF (General Data Format), also a scientific and medical data file format. Lastly, we have MAT format, which supports many data types including signed and unsigned, 8-bit, 16-bit, 32-bit, and 64-bit data types, a special data type that represents MATLAB arrays, Unicode-encoded character data, and data stored in compressed format. To work with these data formats, the MNE and Scipy libraries were used. MNE provides a toolkit for loading and processing biological signals, while Scipy provides algorithms for optimization, integration, interpolation, eigenvalue problems, algebraic equations, differential equations, statistics, and many other tasks. In this case, Scipy was used to load MAT files.

Although we have functions that load the type of data mentioned above, it was still necessary to develop some functions in order to reshape and aggregate the signal in the following format.

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
As long as we test state-of-the-art algorithms we need to have some sort of metrics baseline considering the traditional literature models. In order to accomplish that, before start playing with Deep Learning, we will explore some Traditional Machine Learning algorithms. The idea here is to implement lots of feature extraction methods and try out various combinations of pipelines to understand the better approach for the given datasets. Each dataset will be trained using two strategies, firstly trying only one individual per training session, and secondly passing data from all individuals at once.
In addition, we will choose the more performative pipeline to define as our benchmark.


## 5. Train Convolutional Neural Networks to have a benchmark.
Similar to the previous section, now we will implement three different Convolution Neural Networks and test the data, using the same strategy of training only with one individual data, then training with the whole dataset. The CNNs were selected on the following GitHub repository.

[Army Research Laboratory (ARL) EEGModels project: A Collection of Convolutional Neural Network (CNN) models for EEG signal processing and classification](https://github.com/vlawhern/arl-eegmodels)

    > CNNs
        - EEGNet
        - DeepConvNet
        - ShallowConvNet

Moreover, we will summarize and aggregate the results for further analysis and comparison.

## 6. Traine Transformer-based models

It is still under development...

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
