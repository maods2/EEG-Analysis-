import mne
from pathlib import Path
import numpy as np
import scipy


def load_eeg_data_gdf(directory:str, extension:str):
    label_mapping = {7:0, 8:1, 9:2, 10:3}
    file_name_list = list(Path(directory).glob(extension))
    features, labels =[], []
    for file_name in file_name_list:
        feature, label = read_data_gdf(file_name)
        features.append(feature)
        labels.append(label)
    features = np.array([signal for data in features for signal in data])
    labels = np.array([label_mapping[_label] for data in labels for _label in data])
    return features, labels

def read_data_gdf(path):
    raw=mne.io.read_raw_gdf(path,preload=True,
                            eog=['EOG-left', 'EOG-central', 'EOG-right'])
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.set_eeg_reference()
    raw.filter(l_freq=1,h_freq=45)
    events=mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events[0], event_id=[7,8,9,10],on_missing ='warn',tmin= -0.1, tmax=0.7, preload=True)
    labels=epochs.events[:,-1]
    features=epochs.get_data()
    return features,labels





def load_eeg_data_edf(directory:str, extension:str):
    p = Path(directory)
    file_name_list = [x for x in p.glob(extension) if not x.name.endswith("02.edf") and not x.name.endswith("01.edf") ]
    features, labels =[], []
    for file_name in file_name_list:
        feature, label = read_data_edf(file_name)
        features.append(feature)
        labels.append(label)
        del feature
        del label
        gc.collect()

    features = np.array([signal for data in features for signal in data])
    labels = np.array([_label for data in labels for _label in data])
    return features, labels

def read_data_edf(file_name):
    file_name = str(file_name)

    condition_class0 = ('R03' in file_name) or ('R07' in file_name) or ('R11' in file_name)
    condition_class1 = ('R04' in file_name) or ('R08' in file_name) or ('R12' in file_name)
    condition_class2 = ('R05' in file_name) or ('R09' in file_name) or ('R13' in file_name)
    condition_class3 = ('R06' in file_name) or ('R10' in file_name) or ('R14' in file_name)

    raw = mne.io.read_raw_edf(file_name,preload=True)
    raw.set_eeg_reference()
    events = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events[0], event_id=[2,3],on_missing ='warn')
    features = epochs.get_data()
    del raw
    del events
    del epochs
    gc.collect()
    num_trials = len(features)

    if condition_class0:
        labels = np.full(num_trials, 0)
    elif condition_class1:
        labels = np.full(num_trials, 1)
    elif condition_class2:
        labels = np.full(num_trials, 2)
    elif condition_class3:
        labels = np.full(num_trials, 3)
    
    return features,labels



def read_data_mat(file_path:str):
    m = scipy.io.loadmat(file_path, struct_as_record=True)

    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered

    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    nclasses = len(cl_lab)
    nevents = len(event_onsets)

    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    win = np.arange(int(0.5*sample_rate), int(2.5*sample_rate))

    # Length of the time window
    nsamples = len(win)

    # Print some information
    print('Shape of EEG:', EEG.shape)
    print('Sample rate:', sample_rate)
    print('Number of channels:', nchannels)
    print('Channel names:', channel_names)
    print('Number of events:', nevents)
    print('Event codes:', np.unique(event_codes))
    print('Class labels:', cl_lab)
    print('Number of classes:', nclasses)


    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((len(cl_onsets),nchannels, nsamples))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][i,:,:] = EEG[:, win+onset]
    
    # Some information about the dimensionality of the data (channels x time x trials)
    print('Shape of trials[cl1]:', trials[cl1].shape)
    print('Shape of trials[cl2]:', trials[cl2].shape)
    x =  np.concatenate([trials[cl1], trials[cl2]], axis=0)
    y =  np.concatenate([np.full(len(trials[cl1]),0), np.full(len(trials[cl1]),1)], axis=0)
    return x, y