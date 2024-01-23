# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:17:42 2023

@author: HRGiri
"""

import numpy as np
import pandas as pd
import scipy
import nitime.algorithms as tsa
from datetime import timedelta



symbol = {
    'alpha' : chr(945),
    'beta' : chr(946),
    'gamma' : chr(947),
    'delta' : chr(948),
    'theta' : chr(952),
    'mu' : chr(956),
}

bands = {
    'delta' : (0,4),
    'theta' : (4,8),
    'alpha' : (8,13),
    'mu' : (8,12),
    'beta' : (13,30),
    'beta1': (13,18),
    'beta2': (18,22),
    'beta3': (22,26),
    'beta4': (26,30),
    'gamma': (30,100),
    }


def resample_emg(fp, out_fp=None, time_col=0, start_col=1, end_col=3):
    '''
    Resample EMG recorded from Delsys software.

    Parameters
    ----------
    fp : str
        Filepath of the csv file to be resampled.
        Arranged in time_col emg1_col emg2_col ...
    out_fp : str, optional
        Output filepath. 
        When nothing is supplied, it used the input filepath to generate the output filepath.
    time_col : int, optional
        The time column in the dataframe (in seconds). The default is 0.
    start_col : int, optional
        First column of the EMG readings in the dataframe. The default is 1.
    end_col : int, optional
        First column of the EMG readings in the dataframe. The default is 3.

    Returns
    -------
    None.

    '''
    print("Reading File...")
    emg_df = pd.read_csv(fp)
    print("Resampling...")
    durations =  [timedelta(seconds=x) for x in emg_df.iloc[:,time_col]]
    tdi = pd.TimedeltaIndex(durations)
    emg_df['tdindex'] = tdi
    emg_df = emg_df.set_index('tdindex')
    emg_df = emg_df.iloc[:,start_col:end_col]
    resampled_emg_df = emg_df.resample('L').mean()
    print("Saving...")
    resampled_emg_df = resampled_emg_df.reset_index(drop=True)
    
    if out_fp is None:
        tmp = fp.split('.')
        tmp.insert(1, '_resampled')
        out_fp = tmp[0] + tmp[1] + '.' + tmp[2]
    
    
    resampled_emg_df.to_csv(out_fp, index=False)
    print("Done!")



def dB(x):
  """
  Get values in deciBels

  Parameters
  ----------
  x : ndarray or array-like
    Input array.

  Returns
  -------
  TYPE
    DESCRIPTION.

  """
  return 10 * np.log10(x)


def get_psd(data, method=None, fs=1000, fmin=0, fmax=30, m=1, batch=True):
  """
  Returns the Power Spectral Density from an ndarray of time-series trials.

  Parameters
  ----------
  data : ndarray
    The time-series EEG data segmented into Epochs with the last axis as the sample axis.
  method : str, optional
    The method to calculate PSD. Choose between 'welch' or 'multitaper'. 
    If None, then the default method of mne.Epochs is used. The default is None.
  fs : int, optional
    Sampling Frequency in samples per second. The default is 1000.
  fmin : int, optional
    Lower cutoff frequency. The default is 0.
  fmax : int, optional
    Upper cutoff frequency. The default is 30.
  m : float, optional
    The fraction of number of samples to be taken when calculating PSD using the 'welch' method. 
    The default is 1.
  batch : bool, optional
    Whether processing a batch or single trial. The default is True.

  Raises
  ------
  Exception

  Returns
  -------
  (ndarray, ndarray)
    If method is one of the designated strings, returns a tuple of ndarrays (PSD, frequencies).

  """
  
  Fs = fs

  if method == 'welch':
    nperseg=int(data.shape[-1]*m)
    f, P = scipy.signal.welch(data, Fs, scaling='density', nperseg=nperseg)

  elif method == 'multitaper':
    f, P, nu = tsa.multi_taper_psd(data, Fs=Fs, jackknife=False)
    # print(f"Multitaper spectrum estimation with {int(nu[0][0,0]/2)} DPSS windows")

  else:
    raise Exception('Undefined method. Chose between None, "welch" or "multitaper"')

  P = P[:,:,np.where(f<=fmax)[0]] if batch else P[:,np.where(f<=fmax)[0]]
  f = f[np.where(f<=fmax)[0]]
  P = P[:,:,np.where(f>=fmin)[0]] if batch else P[:,np.where(f>=fmin)[0]]
  f = f[np.where(f>=fmin)[0]]

  return P, f



def bandpass(P, f, band, batch=True):
  """
  Band limits a PSD ndarray to the provided band.

  Parameters
  ----------
  P : ndarray
    The Power Spectral Density to be band limited. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.
  band : (int, int)
    Lower and Upper cutoff frequencies.
  batch : bool, optional
    Whether processing a batch or single trial. The default is True.

  Returns
  -------
  Pb : ndarray
    Band limited PSD.
  fb : ndarray
    Frequencies.

  """
  fmin, fmax = band
  Pb = P[:,:,np.where(f<=fmax)[0]] if batch else P[:,np.where(f<=fmax)[0]]
  fb = f[np.where(f<=fmax)[0]]
  Pb = Pb[:,:,np.where(fb>=fmin)[0]] if batch else Pb[:,np.where(fb>=fmin)[0]]
  fb = fb[np.where(fb>=fmin)[0]]
  return Pb, fb



# Feature Extraction Methods

def get_shannon_entropy(P,batch=True):
  """
  Reutrns the Shannon Entropy of a given Power Spectral Density.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  batch : bool, optional
    Whether processing a batch or single trial. The default is True.

  Returns
  -------
  ndarray
    Shannon Entropy of the given PSD.

  """
  p = np.zeros(P.shape)
  if batch:
    for trial in range(P.shape[0]):
      for ch in range(P.shape[1]):
        p[trial, ch] = P[trial,ch] / P[trial,ch].sum()
  else:
    for ch in range(P.shape[0]):
      p[ch] = P[ch] / P[ch].sum()
  return (p * np.log(p)).sum(axis=-1)



def get_mean_frequency(P, f):
  """
  Mean frequency of a PSD.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.

  Returns
  -------
  ndarray
    Mean Frequency of the PSD.

  """
  return (P*f).sum(axis=-1) / P.sum(axis=-1)



def get_median_frequency(P, f):
  """
  Median frequency of a PSD.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.

  Returns
  -------
  mdfs : ndarray
    Mean Frequency of the PSD.

  """
  cs = np.cumsum(P, axis=-1)
  mdfs = np.zeros(cs.shape[:-1])
  if len(P.shape) == 3:
    for t in range(mdfs.shape[0]):
      for ch in range(mdfs.shape[1]):
        mdfs[t,ch] = f[np.where(cs[t,ch,:]>np.max(cs[t,ch,:])/2)[0][0]]
  else:
    for ch in range(mdfs.shape[0]):
        mdfs[ch] = f[np.where(cs[ch,:]>np.max(cs[ch,:])/2)[0][0]]

  return mdfs



def get_peak_frequency(P, f):
  """
  Peak frequency of a PSD.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.

  Returns
  -------
  ndarray
    Peak Frequency of the PSD.

  """
  return f[P.argmax(axis=-1)]



def get_mean_power(P, f):
  """
  Mean power of a PSD.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.

  Returns
  -------
  ndarray
    Mean power of the PSD.

  """
  return P.sum(axis=-1) / f.size



def get_spectral_moments(P, f, n=0, return_all=False, concat=False):
  """
  Returns the Spectral Moments from a PSD.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.
  n : int, optional
    Order of the Spectral Moment. If 0, returns the Total Power of the PSD. The default is 0.
  return_all : bool, optional
    Whether to return the given order of the spectral moment or all moments upto the given order. 
    The default is False.
  concat : bool, optional
    Whether to concatenate the spectral moments into one ndarray or not. The default is False.

  Returns
  -------
  ndarray or list of ndarray
    Returns ndarray if return_all is False or return_all and concat are both True. 
    Returns a list of ndarrays if return_all is True and concat is False.

  """
  if return_all:
    SMs = []
    for m in range(n + 1):
      SM = (P*(f**m)).sum(axis=-1)
      SMs.append(SM)

    if concat:
      return np.concatenate(SMs, axis=-1)
    else:
      return SMs

  else:
    return (P*(f**n)).sum(axis=-1)
  
  

def get_band_power(P, f, band, batch=True):
  """
  Retruns the total power in a band of frequencies of a PSD.

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.
  band : (int, int)
    Lower and Upper cutoff frequencies.
  batch : bool, optional
    Whether processing a batch or single trial. The default is True.

  Returns
  -------
  ndarray
    Band power of the PSD.

  """
  Pb, fb = bandpass(P, f, band, batch)
  return get_spectral_moments(Pb, fb)



def get_vcf(P, f):
  """
  The Variance in Center Frequency of a PSD

  Parameters
  ----------
  P : ndarray
    Power Spectral Density. The last axis is considered as frequency-axis.
  f : ndarray
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.

  Returns
  -------
  ndarray
    Variance of Centre Frequency of the PSD.

  """
  SMs = get_spectral_moments(P, f, n=2, return_all=True)
  return (SMs[2] / SMs[0]) - ((SMs[1] / SMs[0]) ** 2)



def extract_spectral_features(P=None, f=None, feature_names=None, return_feature_names=False, batch=True):
  """
  Extract Frequency domain features from a given PSD ndarray. 
  Can also be used to return only the feature names.

  Parameters
  ----------
  P : ndarray, optional
    Power Spectral Density. The last axis is considered as frequency-axis. 
    Must be provided if return_feature_names is False. The default is None.
  f : ndarray, optional
    1d array of frequencies associated with the PSD. Should be of the same shape as the last axis of P.
    Must be provided if return_feature_names is False. The default is None.
  feature_names : list or array-like of str, optional
    List of names of features to be extracted. The default is None.
  return_feature_names : bool, optional
    Whether to return only the feature names. The default is False.
  batch : bool, optional
    Whether processing a batch or single trial. The default is True.

  Returns
  -------
  ndarray
    Extracted features of shape (num_trials, num_channels * num_features).

  """
  funcs = {
        'Mean Frequency': get_mean_frequency,
        'Median Frequency': get_median_frequency,
        'Peak Frequency': get_peak_frequency,
        'Mean Power': get_mean_power,
        'Total Power': get_spectral_moments,
        'Spectral Moment 1': lambda P, f: get_spectral_moments(P, f, n=1),
        'Spectral Moment 2': lambda P, f: get_spectral_moments(P, f, n=2),
        'Spectral Moment 3': lambda P, f: get_spectral_moments(P, f, n=3),
        'Variance in Central Frequency': get_vcf,
        'Shannon Entropy': lambda P, f: get_shannon_entropy(P, batch=batch),
        f'Band Power in {symbol["theta"]}': lambda P, f: get_band_power(P, f, bands['theta'], batch),
        f'Band Power in {symbol["alpha"]}': lambda P, f: get_band_power(P, f, bands['alpha'], batch),
        'Skewness': lambda P, f: scipy.stats.skew(P, axis=-1),
        'Kurtosis': lambda P, f: scipy.stats.kurtosis(P, axis=-1),
    }

  shorthands = {
      'MNF' : 'Mean Frequency',
      'MDF' : 'Median Frequency',
      'PKF' : 'Peak Frequency',
      'MNP' : 'Mean Power',
      'TTP' : 'Total Power',
      'SM1' : 'Spectral Moment 1',
      'SM2' : 'Spectral Moment 2',
      'SM3' : 'Spectral Moment 3',
      'VCF' : 'Variance in Central Frequency',
      'ShEn': 'Shannon Entropy',
      'BPT' : f'Band Power in {symbol["theta"]}',
      'BPA' : f'Band Power in {symbol["alpha"]}',
      'skew': 'Skewness',
      'kurt': 'Kurtosis',
  }

  if feature_names is None:
    feature_names = funcs.keys()

  if return_feature_names:
    return feature_names

  features = []
  for name in feature_names:
    if name in shorthands.keys():
      name = shorthands[name]
    features.append(funcs[name](P, f))

  return np.concatenate(features, axis=-1)



# Time-Domain features

def get_mav(X):
  """
  Returns the Mean Absolute Value of time-series signals.

  Parameters
  ----------
  X : ndarray
    Time domain signal. Last axis is the time-axis.

  Returns
  -------
  ndarray
    MAV of X.

  """
  return np.abs(X).sum(axis=-1)/X.shape[-1]




def get_rms(X):
  """
  Returns the Root Mean Square of time-series signals.

  Parameters
  ----------
  X : ndarray
    Time domain signal. Last axis is the time-axis.

  Returns
  -------
  ndarray
    RMS of X.

  """
  return np.sqrt(np.square(X).sum(axis=-1)/X.shape[-1])




def get_logD(X):
  """
  Returns the Logarithm Detector of time-series signals.

  Parameters
  ----------
  X : ndarray
    Time domain signal. Last axis is the time-axis.

  Returns
  -------
  ndarray
    LogD of X.

  """
  return np.exp(np.log10(np.abs(X)).sum(axis=-1)/X.shape[-1])




def get_activity(X):
  """
  Returns the Hjorth Activity of time-series signals.

  Parameters
  ----------
  X : ndarray
    Time domain signal. Last axis is the time-axis.

  Returns
  -------
  ndarray
    Hjorth Activity of X.

  """
  return scipy.stats.tvar(X, axis=-1)




def get_mobility(X):
  """
  Returns the Hjorth Mobility of time-series signals.

  Parameters
  ----------
  X : ndarray
    Time domain signal. Last axis is the time-axis.

  Returns
  -------
  ndarray
    Hjorth Mobility of X.
    
  """
  X_ = np.diff(X, axis=-1)
  return np.sqrt(get_activity(X_)/get_activity(X))




def get_complexity(X):
  """
  Returns the Hjorth Complexity of time-series signals.

  Parameters
  ----------
  X : ndarray
    Time domain signal. Last axis is the time-axis.

  Returns
  -------
  ndarray
    Hjorth Complexity of X.

  """
  X_ = np.diff(X, axis=-1)
  return get_mobility(X_)/get_mobility(X)


def get_mean_temporal_power(X):
  """
  Returns Mean Power of a time domain signal.

  Parameters
  ----------
  X : ndarray
    Time-domain signal with last axis as time axis.

  Returns
  -------
  ndarray
    Mean Power.

  """
  return np.square(X).mean(axis=-1)



def extract_temporal_features(X=None, feature_names=None, return_feature_names=False, batch=True):
  """
  Extract Time domain features from a given time-series ndarray. 
  Can also be used to return only the feature names.


  Parameters
  ----------
  X : ndarray, optional
    Time domain signal. Last axis is the time-axis. The default is None.
  feature_names : list or array-like of str, optional
    List of names of features to be extracted. The default is None.
  return_feature_names : bool, optional
    Whether to return only the feature names. The default is False.
  batch : bool, optional
    Whether processing a batch or single trial. The default is True.

  Returns
  -------
  ndarray
    Extracted features of shape (num_trials, num_channels * num_features).

  """
  funcs = {
        'Mean Absolute Value': get_mav,
        'Root Mean Square': get_rms,
        'Logarithm Detector': get_logD,
        'Hjorth Activity': get_activity,
        'Hjorth Mobility': get_mobility,
        'Hjorth Complexity': get_complexity,
    }

  shorthands = {
      'MAV' : 'Mean Absolute Value',
      'RMS' : 'Root Mean Square',
      'LOGD': 'Logarithm Detector',
      'HA'  : 'Hjorth Activity',
      'HM'  : 'Hjorth Mobility',
      'HC'  : 'Hjorth Complexity',
  }

  if feature_names is None:
    feature_names = funcs.keys()

  if return_feature_names:
    return feature_names

  features = []
  for name in feature_names:
    if name in shorthands.keys():
      name = shorthands[name]
    features.append(funcs[name](X))

  return np.concatenate(features, axis=-1)


def common_average_reference(data, ch_axis=1):
  """
  Performs Common Average Referencing on multi-channel EEG data.

  Parameters
  ----------
  data : ndarray
    EEG data with one of the axis as channel axis.
  ch_axis : int, optional
    Channel axis. The default is 1.

  Returns
  -------
  ndarray
    Common average referenced EEG data.

  """
  mean = np.mean(data, axis=ch_axis)
  mean = np.expand_dims(mean, ch_axis)
  mean = np.concatenate([mean] * data.shape[ch_axis], axis=ch_axis)
  return data - mean



def pick_channels(data, picks=None, ch_names=None, ch_axis=1):
  """
  Pick a subset of channels in an ndarray.

  Parameters
  ----------
  data : ndarray
    EEG data with one of the axis as channel axis.
  picks : list, optional
    List of channel names to be picked.
    If None, the array is returned as it is. The default is None.
  ch_names : list, optional
    List of all channel names present in the array. The default is None.
  ch_axis : int, optional
    Channel axis. The default is 1.

  Raises
  ------
  Exception
    Raised if neither of picks or ch_names is provided.

  Returns
  -------
  ndarray
    EEG data with a subset of channels.

  """
  if picks is None:
    return data
  if ch_names is None:
    raise Exception("Please provide the list of all channel names.")
  ch_indices = [ch_names.index(pick) for pick in picks]
  return np.take(data, ch_indices, axis=ch_axis)
