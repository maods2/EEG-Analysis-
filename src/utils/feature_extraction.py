import emd
from dtw import dtw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import	antropy as ant
import scipy
import numpy as np
import pywt
from sklearn.decomposition import PCA

def dtw_auxiliar(x, y):
  return dtw(x, y, keep_internals=True).distance # Dynamic Time Warp

def dtw_fast(x, y):
  return fastdtw(x, y)[0]

def emd_transform(input_narray, params={}):
  x_emd = np.zeros(np.shape(input_narray))
  for i in range(len(x_emd)):
    emd_result = emd.sift.sift(input_narray[i]) #Compute Intrinsic Mode Functions
    x_emd[i] = np.sum(emd_result[:,3:], axis=1)
  return x_emd

def bandpass_transform(input_narray, params={}):
  fs = params['fs'] if 'fs' in params else 250
  low_freq = 8
  high_freq = 15
  a, b = scipy.signal.iirfilter(6, [low_freq/(fs/2.0), high_freq/(fs/2.0)])
  trials_filt= scipy.signal.filtfilt(a, b, input_narray, axis=-1)
  return trials_filt


def psd_transform(input_narray, params={}):
  fs = params['fs'] if 'fs' in params else 250
  f, psd = scipy.signal.welch(input_narray, fs=fs)
  return psd

def standard_scaler_transform(input_narray, params={}):
  scaler = StandardScaler()
  data = scaler.fit_transform(input_narray)
  return data
  
def pca_transform(input_narray, params={}):
  # print(np.shape(input_narray))
  pca = PCA(n_components=2)
  data = pca.fit_transform(input_narray)
  return data  

def wavelet_transform(input_narray, params={}):
  fs = params['fs'] if 'fs' in params else 250
  dec_wave_lvs = pywt.wavedec(input_narray, 'coif1', level=2) 
  return dec_wave_lvs[0]

def time_domain_features_transform(input_narray, params={}):
  fs = params['fs'] if 'fs' in params else 250
  x_features = time_domain_features(input_narray, fs)
  return x_features


def frequency_domain_features_transform(input_narray, params={}):
  fs = params['fs'] if 'fs' in params else 250
  features_number = frequency_domain_features(input_narray[0], fs).shape[-1]
  x_features = np.zeros((input_narray.shape[0], features_number))
  for i in range(len(x_features)):
    x_features[i] = frequency_domain_features(input_narray[i], fs)
  return x_features

# Features copied from Python Audio Analysis Library
# https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioFeatureExtraction.py
def spectral_centroid_and_spread(X, fs):
    ''' Spectral centroid (SC) measures the shape of the spectrum of a signals. 
    A higher value of SC corresponds to more energy of the signal being concentrated 
    within higher frequencies. Basically, it measures the spectral shape and position 
    of the spectrum.  
    '''
    EPS = 0.00000001
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))
    Xt = X.copy()
    Xt = Xt / Xt.max()
    numerator = np.sum(ind * Xt)
    denominator = np.sum(Xt) + EPS
    centroid = (numerator / denominator)
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / denominator)
    # Normalize:
    centroid = centroid / (fs / 2.0)
    spread = spread / (fs / 2.0)
    return centroid, spread


def spectral_rolloff(X, c):
    ''' Spectral rolloff point in Hz, returned as a scalar, vector, or matrix. 
    Each row of rolloffPoint corresponds to the spectral rolloff point of a 
    window of x .
    '''
    EPS = 0.00000001
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position 
    # where the respective spectral energy is equal to c*totalEnergy
    CumSum = np.cumsum(X ** 2) + EPS
    [a, ] = np.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return mC

def peaks(x):
    data_max = np.amax(x)
    data_index, = np.where(x == np.amax(x))
    peak_value, peak_frequency = data_max, int(data_index) + 1
    return peak_value, peak_frequency

def mean(x):
  return np.mean(x, axis=-1)

def var(x):
  return np.var(x, axis=-1)

def std(x):
  return np.std(x, axis=-1)  

# def skewness(x):
#   return np.array(stats.skewness(x))

# def kurtosis(x):
#   return np.array(stats.kurtosis(x))

def ptp(x):
  return np.ptp(x, axis=-1)

def argmin(x):
  return np.argmin(x, axis=-1)

def argmax(x):
  return np.argmax(x, axis=-1)

def abs_diff_signal(x):
  return np.nansum(np.abs(np.diff(x, axis=-1)), axis=-1)

def rms(x):
  return np.sqrt(np.nanmean(x**2, axis=-1))

def time_spectral_entropy(x, fs):  # Applied for time domain because has an inner fft calculation 
  return ant.spectral_entropy(x, sf=fs, method='welch', normalize=True)

def frequency_spectral_entropy(psd, normalize=True, axis=-1):
  psd_norm = psd / psd.sum(axis=axis, keepdims=True)
  se = -(psd_norm * np.log2(psd_norm)).sum(axis=axis)
  if normalize:
    se /= np.log2(psd_norm.shape[axis])
  return se


def time_domain_features(x, fs):
  ax=-1
  mean_result = mean(x)
  var_result = var(x)
  skewness_result = scipy.stats.skew(x, axis=ax)
  kurtosis_result = scipy.stats.kurtosis(x, axis=ax)
  ptp_result = ptp(x)
  abs_diff_signal_result = abs_diff_signal(x)
  rms_result = rms(x)
  time_spectral_entropy_result = time_spectral_entropy(x, fs)
  return np.round(np.hstack([
              mean_result,
              var_result,
              skewness_result,
              kurtosis_result,
              ptp_result,
              abs_diff_signal_result,
              rms_result,
              time_spectral_entropy_result
              ]),4)

def frequency_domain_features(x, fs):
  ax=-1
  # centroid, spread = spectral_centroid_and_spread(x, fs)
  # spectral_rolloff_result = spectral_rolloff(x, centroid)
  peak_value, peak_frequency = peaks(x)
  # argmin_result = argmin(x)
  # argmax_result = argmax(x)
  frequency_spectral_entropy_result = frequency_spectral_entropy(x)
  mean_result = mean(x)
  var_result = var(x)
  std_result = std(x)
  skewness_result = scipy.stats.skew(x, axis=ax)
  kurtosis_result = scipy.stats.kurtosis(x)
  # ptp_result = ptp(x)
  # abs_diff_signal_result = abs_diff_signal(x)
  rms_result = rms(x)

 
  return  np.round(np.hstack([
            # spectral_rolloff_result,
            # centroid,
            # spread,
            peak_value,
            peak_frequency,
            # argmin_result,
            # argmax_result,
            frequency_spectral_entropy_result,
            mean_result,
            var_result,
            std_result,
            skewness_result,
            kurtosis_result,
            # ptp_result,
            # abs_diff_signal_result,
            rms_result,
          ]),4)
