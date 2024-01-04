# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:43:08 2023

@author: HRGiri
"""

import glob
import os

import mne

import numpy as np
import scipy
import nitime.algorithms as tsa


annotations_dict = {'Comment/Blinking' : 'Blink',
                    'Comment/Hand Close' : 'HC',
                    'Comment/Hand open' : 'HO',
                    'Comment/Imagine Open' : 'IO',
                    'Comment/Imagine close' : 'IC',
                    'Comment/Left' : 'LH',
                    'Comment/Ready' : 'Ready',
                    'Comment/Rest' : 'Rest',
                    'Comment/Right' : 'RH'}

def load_raws(subject_name, preload=True, in_drive=False, dir_path=None):
  """
  Load mne raw objects from vhdr files.

  Parameters
  ----------
  subject_name : str
    Name of the subject folder.
  preload : bool, optional
    Whether to load data in memory as well. The default is True.
  in_drive : bool, optional
    Whether accessing files from Google Drive. The default is False.
  dir_path : str, optional
    Custom path for the data directory. The default is None.

  Returns
  -------
  raws : list
    List of mne.io.Raw objects.

  """
  
  # directory where the vhdr files are located
  if dir_path is None:
    if in_drive:
      dir_path = '/content/drive/Shareddrives/EEG Drive/Data/Brainvision/'
    else:
      dir_path = "D:\\EEG Data\\Raw EEG\\Hand OpenClose Protocol\\"
  directory = dir_path + subject_name
  
  # list all vhdr files in the directory
  edf_files = glob.glob(os.path.join(directory, '*.vhdr'))
  # print(edf_files)

  # use list comprehension to read all files
  raws = [mne.io.read_raw_brainvision(f, preload = preload) for f in edf_files]

  # Concatenate all raws
  # raw = mne.io.concatenate_raws(raws)

  # Rename Annotations to something simpler
  for raw in raws:
    raw.annotations.rename(annotations_dict)

  return raws


def get_epochs(raw, target=None, tmin=0, tmax=3, return_events=False):
  """
  Extract epochs (mne.Epochs) from raw (mne.io.Raw) files.
  Can also be used to extract events from the epochs

  Parameters
  ----------
  raw : mne.io.Raw
    Mne raw object from which to extract mne epochs.
  target : str, optional
    Event name to be accessed. Can be used to extract multiple events by mne Epochs convention.
    None returns all events.
    The default is None.
  tmin : float, optional
    Start time (in s) of the event (relative to where marker appears). The default is 0.
  tmax : float, optional
    End time (in s) of the event (relative to where the marker appears). The default is 3.
  return_events : bool, optional
    Whether to only return the events. The default is False.

  Returns
  -------
  mne.Epochs
    Epochs object of the target events.
  
  Also returns
  -------
  ndarray(num_events,3)
    Events of the target Epochs. 
    1st column contains time sample, 3rd column contains event_id of the event.

  """
  events, event_dict = mne.events_from_annotations(raw)
  event_dict['Real/Left/Close'] = 99
  event_dict['Real/Left/Open'] = 88
  event_dict['Real/Right/Close'] = 9
  event_dict['Real/Right/Open'] = 8
  event_dict['Imagine/Left/Close'] = 77
  event_dict['Imagine/Left/Open'] = 66
  event_dict['Imagine/Right/Close'] = 7
  event_dict['Imagine/Right/Open'] = 6

  isRight = False
  for i, event in enumerate(events):
    if event[-1] == event_dict['LH']:
      isRight = False
    elif event[-1] == event_dict['RH']:
      isRight = True
    elif event[-1] == event_dict['HC']:
      events[i][-1] = event_dict['Real/Right/Close'] if isRight else event_dict['Real/Left/Close']
    elif event[-1] == event_dict['HO']:
      events[i][-1] = event_dict['Real/Right/Open'] if isRight else event_dict['Real/Left/Open']
    elif event[-1] == event_dict['IC']:
      events[i][-1] = event_dict['Imagine/Right/Close'] if isRight else event_dict['Imagine/Left/Close']
    elif event[-1] == event_dict['IO']:
      events[i][-1] = event_dict['Imagine/Right/Open'] if isRight else event_dict['Imagine/Left/Open']

  to_be_deleted = ['LH', 'RH', 'HC', 'HO', 'IC', 'IO']
  for key in event_dict.keys():
    if key.split('/')[0] == 'New Segment':
      to_be_deleted.append(key)

  for key in to_be_deleted:
    del event_dict[key]

  if return_events:
    return events, event_dict

  # baseline = (None, 0)
  # baseline = None

  # Create epochs
  epochs = mne.Epochs(raw,
                      events, event_dict,
                      tmin, tmax,
                      baseline=None,
                      preload=True
                    )
  return epochs if target is None else epochs[target]



def standardize_labels(labels, mapping=None):
  """
  Converts target labels into whole number labels, or according to mapping (if provided)

  Parameters
  ----------
  labels : list or array-like
    1d array of labels.
  mapping : dict, optional
    Mapping of source labels to target labels. 
    If None, the mapping is done sequentially in the order the labels appear in the source list. 
    The default is None.

  Returns
  -------
  ndarray (1d)
    Target labels.

  """
  if mapping is None:
    mapping = {}
  code = 0
  Y = []
  for label in labels:
    if label not in mapping.keys():
      mapping[label] = code
      code += 1
    Y.append(mapping[label])
  return np.array(Y)


def get_psd(epochs, method=None, isData=False, fs=1000, fmin=0, fmax=30, picks='eeg', m=1):
  """
  Returns the Power Spectral Density from a mne.Epochs object.

  Parameters
  ----------
  epochs : mne.Epochs or ndarray
    The time-series EEG data segmented into Epochs.
  method : str, optional
    The method to calculate PSD. Choose between 'welch' or 'multitaper'. 
    If None, then the default method of mne.Epochs is used. The default is None.
  isData : bool, optional
    Whether epochs is ndarray or mne.Epochs object. The default is False.
  fs : int, optional
    Sampling Frequency in samples per second. The default is 1000.
  fmin : int, optional
    Lower cutoff frequency. The default is 0.
  fmax : int, optional
    Upper cutoff frequency. The default is 30.
  picks : list or str, optional
    If list, list of channels to choose from the mne.Epochs object. 
    For str, refer to mne documentation. 
    The default is 'eeg'.
  m : float, optional
    The fraction of number of samples to be taken when calculating PSD using the 'welch' method. 
    The default is 1.

  Raises
  ------
  Exception

  Returns
  -------
  TYPE
    If method is None, returns a mne object.
  TYPE
    If method is one of the designated strings, returns a tuple of ndarrays (PSD, frequencies).

  """
  
  if method is None:
    if isData:
        raise Exception('isData is True. If you intended to pass timeseries data, then choose either "welch" or "multitaper" as method. If you intented to pass an Epochs object, then please set isData to False')
    return epochs.set_eeg_reference(ref_channels='average').compute_psd(fmax=epochs.info['lowpass'], picks=picks).get_data(return_freqs=True)

  else:
    # data = epochs.set_eeg_reference(ref_channels='average').get_data(picks=picks)
    data = epochs.get_data(picks=picks) if not isData else epochs
    Fs = fs if isData else epochs.info['sfreq']

    if method == 'welch':
      nperseg=int(data.shape[-1]*m)
      f, P = scipy.signal.welch(data, Fs, scaling='density', nperseg=nperseg)

    elif method == 'multitaper':
      f, P, nu = tsa.multi_taper_psd(data, Fs=Fs, jackknife=False)
      # print(f"Multitaper spectrum estimation with {int(nu[0][0,0]/2)} DPSS windows")

    else:
      raise Exception('Undefined method. Chose between None, "welch" or "multitaper"')

    P = P[:,:,np.where(f<=fmax)[0]] if not isData else P[:,np.where(f<=fmax)[0]]
    f = f[np.where(f<=fmax)[0]]
    P = P[:,:,np.where(f>=fmin)[0]] if not isData else P[:,np.where(f>=fmin)[0]]
    f = f[np.where(f>=fmin)[0]]

    return P, f