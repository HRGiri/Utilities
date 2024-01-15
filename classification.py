# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:41:45 2024

@author: HRGiri
"""
from time import sleep, time

import numpy as np

from mneUtils import get_epochs, standardize_labels
from signalProcessing import get_psd, extract_spectral_features, extract_temporal_features

# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def extract_train_features(raw, choice=None,slice_start_times=np.array([0]), slice_length=7, window_length=3, time_interval = 1, picks=None, concatenate=True):
  """
  Extracts spectral and temporal features for training.

  Parameters
  ----------
  raw : mne.io.Raw
    Raw object in which time-series data is contained.
  choice : str, optional
    The event label(s) from which to extract features. If None, extracts features from all events. 
    The default is None.
  slice_start_times : ndarray (1d), optional
    The starting points of a slice in s. The default is np.array([0]).
  slice_length : int, optional
    The slice length in s. The default is 3.
  window_length : int, optional
    The length of each window within a slice in s. The default is 3.
  time_interval : int, optional
    Time interval between the start times of two adjacent windows in s. The default is 1.
  picks : list of str, optional
    List of channels to be picked. If None, 9 channels above the sensori-motor cortex are picked.
    The default is None.
  concatenate : bool, optional
    Whether to concatenate the features of different windows together or keep them separate. 
    Keeping them separate can be helpful when a temporal sequence of features is required. 
    The default is True.

  Returns
  -------
  ndarray
    Extracted features with shape (n_trials, n_windows * n_channels * n_features) if number of slices = 1.
    Extracted features with shape (n_slices * n_trials, n_windows * n_channels * n_features) if number of slices > 1.

  """
  if picks is None:
    picks = picks = ['F3','C3','P3','F4','C4','P4','Fz','Cz','Pz']
  X = []
  for slice_start in slice_start_times:
    all_features = []
    epochs = get_epochs(raw, target=choice, tmin=slice_start, tmax=slice_start + slice_length)
    start_times = np.arange(slice_start, slice_start + slice_length - window_length + time_interval, time_interval)
    for tmin in start_times:
      crop = epochs.copy().crop(tmin,tmin + window_length)
      x = crop.get_data(picks=picks)
      P, f = get_psd(crop, method='multitaper')
      fd_features = extract_spectral_features(P, f)
      td_features = extract_temporal_features(x)
      features = np.concatenate([fd_features,td_features], axis=-1)
      all_features.append(features)
      if concatenate:
        features = np.concatenate(all_features, axis=-1)
      else:
        features = np.array(all_features)
    X.append(features)
  x = np.array(X)
  if slice_start_times.size == 1:
    return x.reshape((x.shape[1], x.shape[2]))
  return x.reshape((x.shape[0] * x.shape[1],x.shape[2]),order='F')



def get_train_labels(epochs, n_slices = 1, return_groups=True, mapping={99:0,77:0,88:1,66:1,9:0,7:0,8:1,6:1}):
  """
  Returns standardized labels for epochs.

  Parameters
  ----------
  epochs : mne.Epochs
    Epochs from which labels are to be extracted.
  n_slices : int, optional
    Number of slices. Number of copies of labels to be extracted. The default is 1.
  return_groups : bool, optional
    Whether to return groups from which a gouped cross validation can be done. 
    The default is True.

  Returns
  -------
  ndarray or (ndarray, ndarray)
    If return_groups is True, returns a tuple of the labels and the groups associated with it.
    Otherwise, only the labels is returned.

  """
  Y = standardize_labels(epochs.events[:,2], mapping={99:0,77:0,88:1,66:1,9:0,7:0,8:1,6:1})
  n_trials = Y.shape[0]
  Y = np.array([list(Y)] * n_slices)
  Y = Y.reshape((n_slices * Y.shape[1],),order='F')
  if not return_groups:
    return Y
  else:
    groups = np.array([list(range(n_trials//2))] * n_slices * 2).reshape((Y.shape[0],), order='F')
    return Y, groups
  
  
  
def create_classifier(classifier = 'svm'):
  """
  Creates a sklearn pipeline for hyperparameter optimization with minmax scaling before classification.

  Parameters
  ----------
  classifier : str, optional
    Classifier to be used. 
    Choose from 'svm', 'logreg', 'rf', 'adaboost', 'knn'. 
    The default is 'svm'.

  Returns
  -------
  pipe : sklearn.pipeline.Pipeline
    Pipeline object with MinMaxScaler and selected classifier.
  param_grid_pipeline : dict
    Common parameters associated with the selected classifier for hyperparameter optimization.

  """
  scaler = MinMaxScaler()
  if classifier == 'svm':
    clf = SVC(probability=True)
    param_grid = {
      'C': [0.1, 1, 10],
      'kernel': ['linear', 'rbf', 'poly'],
      'gamma': [0.1, 1, 10]
    }
  elif classifier == 'logreg':
    clf = LogisticRegression(solver='liblinear')
    param_grid = {
        'penalty' : ['l1', 'l2'],
        'C' : np.logspace(-4, 4, 20),
        }
  
  elif classifier == 'rf':
    # Assemble a classifier
    clf = RandomForestClassifier()
    
    # Define the parameter grid
    param_grid = {
        # 'n_estimators': [100, 200, 300],
        # 'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
  
  elif classifier == 'adaboost':
    clf = AdaBoostClassifier()
    param_grid = {}
    param_grid['n_estimators'] = [10, 50, 100, 200]
    param_grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
  
  elif classifier == 'knn':
    clf = KNeighborsClassifier()
    param_grid = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
  # elif classifier == 'ANN':
  #   clf = KerasClassifier(
  #       ANN,
  #       loss='binary_crossentropy',
  #       hidden_units = [128,32,8],
  #       epochs=10,
  #       batch_size=20,
  #       metrics='accuracy',
  #       validation_split=0.1,
  #   )

  #   # Define the parameter grid
  #   param_grid = {
  #       # 'epochs': [50,100],
  #       # 'optimizer': ['adam','rmsprop'],
  #       # 'activation': ['relu', 'tanh'],
  #       'hidden_units': [
  #           [64, 32],
  #           [64, 16],
  #           # [32, 32],
  #           # [32,16],
  #           # [64,32,16],
  #           # [128,32,8],
  #           # [64,8],
  #           # [64,16,16,8],
  #           ],
  #   }
  pipe = Pipeline([
    ('scaler', scaler),
    ('classifier', clf)
    ])
  param_grid_pipeline = {}
  for param in param_grid.keys():
    param_grid_pipeline['classifier__' + param] = param_grid[param]

  return pipe, param_grid_pipeline



def simulate_real_time(
    raw, 
    clf, 
    targets=[['Real/Right/Close']], 
    picks=None,
    label_id = {0: 'Close', 1: 'Open'}, 
    eval_time_limit = 3, 
    realtime = False, 
    window_length = 3, 
    time_interval = 1, 
    n_windows = 5
    ):
  """
  Simulates an mne.io.Raw object wrt time for prediction using trained classifiers.

  Parameters
  ----------
  raw : mne.io.Raw
    Raw object to simulate.
  clfs : dict
    Dictionary of trained classifiers.
  targets : list of lists, optional
    Event labels to be used for prediction. The default is [['Real/Right/Close']].
  picks : list of str, optional
    List of channels to be picked. If None, 9 channels above the sensori-motor cortex are picked.
    The default is None.
  label_id : dict, optional
    Labels to be used for prediction. The default is {0: 'Close', 1: 'Open'}.
  eval_time_limit : float, optional
    Time limit within which correct prediction is to be obtained. The default is 3.
  realtime : bool, optional
    Whether to simulate in Real-Time. The default is False.
  window_length : int, optional
    Length of the window to be used for feature extraction. The default is 3.
  time_interval : int, optional
    Time interval between the start times of two adjacent windows in s. The default is 1.
  n_windows : int, optional
    Number of consecutive windows to extract features from. The default is 5.

  Returns
  -------
  accuracy : dict
    Classification accuracies of different labels.
  avg_eval_time : dict
    Evaluation times of different labels.

  """
  if picks is None:
    picks = picks = ['F3','C3','P3','F4','C4','P4','Fz','Cz','Pz']
  eval_window_open = [False] * len(targets)
  eval_start_time = [0] * len(targets)
  true_count = [0] * len(targets)
  pred_count = [0] * len(targets)
  accuracy = [0] * len(targets)
  avg_eval_time = [0] * len(targets)
  data = raw.get_data(picks=picks)
  events, event_dict = get_epochs(raw, return_events=True)
  event_codes = {v: k for k, v in event_dict.items()}
  event = 0
  offset = 0
  start_time = time() - offset
  code = ""
  feature_vector = []
  n_features = len(list(extract_temporal_features(return_feature_names=True))) + len(list(extract_temporal_features(return_feature_names=True)))
  features_per_window = len(picks) * n_features
  max_features = n_windows * features_per_window
  for i, t in enumerate(raw.times):
    if t < offset:
      continue
    if i % int(time_interval * raw.info['sfreq']) == 0:
      # output.clear(output_tags='time')
      # output.clear(output_tags='code')
      # output.clear(output_tags='features')
      # output.clear(output_tags='prediction')
      # if realtime:
      #     # Line 1
      #     print(f"Time:\t{t:0.1f} %")
      #   # output_message(f"Time:\t{t:0.1f} %", 'time')
      # else:
      #     print(f"Processed:\t{100 * t / raw.times[-1]:0.1f} %")
      #   # output_message(f"Processed:\t{100 * t / raw.times[-1]:0.1f} %", 'time')
      # print(f"\nMarker: {code}")  # Line 3
      #   # output_message(f"\n\nMarker: {code}", 'code')
      # for idx in range(len(targets)):
      #     print(f"Accuracy {label_id[idx]}: {accuracy[idx]:0.2f} %")  # Line 4-5
        # output_message(f"\nAccuracy {label_id[idx]}: {accuracy[idx]:0.2f} %", 'prediction')
        # output_message(f"\nAvg Eval Time {label_id[idx]}: {avg_eval_time[idx]/pred_count[idx]:0.2f} %", 'prediction')

      time_begin = time()
      x = data[:,i:i+int(window_length * raw.info['sfreq'])]
      P, f = get_psd(x, method='multitaper', isData=True)
      fd_features = extract_spectral_features(P, f, batch=False)
      td_features = extract_temporal_features(x)
      features = np.concatenate([fd_features, td_features], axis=-1)
      feature_vector.extend(list(features))

      if len(feature_vector) > max_features:
        del feature_vector[:features_per_window]
        X = np.array(feature_vector).reshape(1,-1)
        y = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        # print(y, proba)

        for idx in range(len(targets)):
          # Evaluate prediction
          if eval_window_open[idx]:
            eval_time = t - eval_start_time[idx]
            if eval_time > eval_time_limit:
              eval_window_open[idx] = False
              # print(f"Failed to predict correctly in {eval_time_limit} s!")   # Line 6 (conditional)
              # output_message(f"\nFailed to predict correctly in {eval_time_limit} s!", 'prediction')
              accuracy[idx] = 100 * pred_count[idx] / true_count[idx]
              # output_message(f"Accuracy: {accuracy:0.2f} %", 'prediction')
            else:
              # Comment this out for per label evaluation
              true_count[idx] += 1
              # Check for true prediction
              if y == idx:
                pred_count[idx] += 1
                # Comment this out for per prediction evaluation
                # eval_window_open[idx] = False
                # avg_eval_time[idx] += eval_time
                # Comment this out for per label evaluation
                avg_eval_time[idx] += time() - time_begin
                # print(f"Correctly predicted in {eval_time:0.2f} s!")  # Line 6 (conditional)
                # output_message(f"\nCorrectly predicted in {eval_time:0.2f} s!", 'prediction')
                accuracy[idx] = 100 * pred_count[idx] / true_count[idx]
                # output_message(f"Accuracy: {accuracy:0.2f} %", 'prediction')
                # print(f"Accuracy: {accuracy:0.2f} %")
        # time_now = time()
        # Line 8
        # print(f"\nPredicted: {label_id[y]} ({100 * proba[y]:0.2f} % confidence)\nTime Taken: {int((time_now - time_begin)*1000)} ms")
        # output_message(f"\n\nPredicted: {label_id[y]} ({100 * proba[y]:0.2f} % confidence)\nTime Taken: {int((time_now - time_begin)*1000)} ms", 'features')
      if realtime:
        sleep(abs(time() - start_time - t))
    if i == events[event,0]:
      try:
        code = event_codes[events[event,2]]
        for idx, target in enumerate(targets):
          if code in target:
            eval_window_open[idx] = True
            eval_start_time[idx] = t
            # Comment this out for per prediction evaluation
            # true_count[idx] += 1
      except KeyError:
        pass

      event += 1
      if event == events.shape[0]:
        break
  for idx in range(len(targets)):
    accuracy[idx] = 100 * pred_count[idx] / true_count[idx]
    avg_eval_time[idx] = avg_eval_time[idx] / pred_count[idx] if pred_count[idx] != 0 else 0
  return accuracy, avg_eval_time




def simulate_real_time_stack(
    raw, 
    clfs, 
    targets=[['Real/Right/Close']], 
    picks=None, 
    label_id = {0: 'Close', 1: 'Open'}, 
    eval_time_limit = 3, 
    realtime = False, 
    window_length = 3, 
    time_interval = 1, 
    n_windows = 5
    ):
  """
  Simulates an mne.io.Raw object wrt time for prediction using trained classifiers.
  For Stacked classifiers.

  Parameters
  ----------
  raw : mne.io.Raw
    Raw object to simulate.
  clfs : dict
    Dictionary of trained classifiers.
  targets : list of lists, optional
    Event labels to be used for prediction. The default is [['Real/Right/Close']].
  picks : list of str, optional
    List of channels to be picked. If None, 9 channels above the sensori-motor cortex are picked.
    The default is None.
  label_id : dict, optional
    Labels to be used for prediction. The default is {0: 'Close', 1: 'Open'}.
  eval_time_limit : float, optional
    Time limit within which correct prediction is to be obtained. The default is 3.
  realtime : bool, optional
    Whether to simulate in Real-Time. The default is False.
  window_length : int, optional
    Length of the window to be used for feature extraction. The default is 3.
  time_interval : int, optional
    Time interval between the start times of two adjacent windows in s. The default is 1.
  n_windows : int, optional
    Number of consecutive windows to extract features from. The default is 5.

  Returns
  -------
  accuracy : dict
    Classification accuracies of different labels.
  avg_eval_time : dict
    Evaluation times of different labels.

  """
  if picks is None:
    picks = picks = ['F3','C3','P3','F4','C4','P4','Fz','Cz','Pz']
  eval_window_open = [False] * len(targets)
  eval_start_time = [0] * len(targets)
  true_count = [0] * len(targets)
  pred_count = [0] * len(targets)
  accuracy = [0] * len(targets)
  avg_eval_time = [0] * len(targets)
  data = raw.get_data(picks=picks)
  events, event_dict = get_epochs(raw, return_events=True)
  event_codes = {v: k for k, v in event_dict.items()}
  event = 0
  offset = 0
  start_time = time() - offset
  code = ""
  feature_vector = []
  n_features = len(list(extract_spectral_features(return_feature_names=True))) + len(list(extract_temporal_features(return_feature_names=True)))
  features_per_window = len(picks) * n_features
  max_features = n_windows * features_per_window
  for i, t in enumerate(raw.times):
    if t < offset:
      continue
    if i % int(time_interval * raw.info['sfreq']) == 0:
      # output.clear(output_tags='time')
      # output.clear(output_tags='code')
      # output.clear(output_tags='features')
      # output.clear(output_tags='prediction')
      # if realtime:
      #     # Line 1
      #     print(f"Time:\t{t:0.1f} %")
      #   # output_message(f"Time:\t{t:0.1f} %", 'time')
      # else:
      #     print(f"Processed:\t{100 * t / raw.times[-1]:0.1f} %")
      #   # output_message(f"Processed:\t{100 * t / raw.times[-1]:0.1f} %", 'time')
      # print(f"\nMarker: {code}")  # Line 3
      #   # output_message(f"\n\nMarker: {code}", 'code')
      # for idx in range(len(targets)):
      #     print(f"Accuracy {label_id[idx]}: {accuracy[idx]:0.2f} %")  # Line 4-5
        # output_message(f"\nAccuracy {label_id[idx]}: {accuracy[idx]:0.2f} %", 'prediction')
        # output_message(f"\nAvg Eval Time {label_id[idx]}: {avg_eval_time[idx]/pred_count[idx]:0.2f} %", 'prediction')

      time_begin = time()
      x = data[:,i:i+int(window_length * raw.info['sfreq'])]
      P, f = get_psd(x, method='multitaper', isData=True)
      fd_features = extract_spectral_features(P, f, batch=False)
      td_features = extract_temporal_features(x)
      features = np.concatenate([fd_features, td_features], axis=-1)
      feature_vector.extend(list(features))

      if len(feature_vector) > max_features:
        del feature_vector[:features_per_window]
        X = np.array(feature_vector).reshape(1,-1)
        y_preds = {}
        for clf_name in clfs:
            clf = clfs[clf_name]
            if clf_name == 'Stack':
                continue
            else:
                y_pred = clf.predict(X)
                y_preds[clf_name] = y_pred
        
        X_stack = np.column_stack(tuple(y_preds.values()))
        y = clfs['Stack'].predict(X_stack)[0]
            # proba = clf.predict_proba(X)[0]
        # print(y, proba)

        for idx in range(len(targets)):
          # Evaluate prediction
          if eval_window_open[idx]:
            eval_time = t - eval_start_time[idx]
            if eval_time > eval_time_limit:
              eval_window_open[idx] = False
              # print(f"Failed to predict correctly in {eval_time_limit} s!")   # Line 6 (conditional)
              # output_message(f"\nFailed to predict correctly in {eval_time_limit} s!", 'prediction')
              accuracy[idx] = 100 * pred_count[idx] / true_count[idx]
              # output_message(f"Accuracy: {accuracy:0.2f} %", 'prediction')
            else:
              # Comment this out for per label evaluation
              true_count[idx] += 1
              # Check for true prediction
              if y == idx:
                pred_count[idx] += 1
                # Comment this out for per prediction evaluation
                # eval_window_open[idx] = False
                # avg_eval_time[idx] += eval_time
                # Comment this out for per label evaluation
                avg_eval_time[idx] += time() - time_begin
                # print(f"Correctly predicted in {eval_time:0.2f} s!")  # Line 6 (conditional)
                # output_message(f"\nCorrectly predicted in {eval_time:0.2f} s!", 'prediction')
                accuracy[idx] = 100 * pred_count[idx] / true_count[idx]
                # output_message(f"Accuracy: {accuracy:0.2f} %", 'prediction')
                # print(f"Accuracy: {accuracy:0.2f} %")
        # time_now = time()
        # Line 8
        # print(f"\nPredicted: {label_id[y]} ({100 * proba[y]:0.2f} % confidence)\nTime Taken: {int((time_now - time_begin)*1000)} ms")
        # output_message(f"\n\nPredicted: {label_id[y]} ({100 * proba[y]:0.2f} % confidence)\nTime Taken: {int((time_now - time_begin)*1000)} ms", 'features')
      if realtime:
        sleep(abs(time() - start_time - t))
    if i == events[event,0]:
      try:
        code = event_codes[events[event,2]]
        for idx, target in enumerate(targets):
          if code in target:
            eval_window_open[idx] = True
            eval_start_time[idx] = t
            # Comment this out for per prediction evaluation
            # true_count[idx] += 1
      except KeyError:
        pass

      event += 1
      if event == events.shape[0]:
        break
  for idx in range(len(targets)):
    accuracy[idx] = 100 * pred_count[idx] / true_count[idx]
    avg_eval_time[idx] = avg_eval_time[idx] / pred_count[idx] if pred_count[idx] != 0 else 0
  return accuracy, avg_eval_time