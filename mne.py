# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:43:08 2023

@author: HRGiri
"""

import glob
import os

import mne


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