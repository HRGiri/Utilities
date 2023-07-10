# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:17:42 2023

@author: HRGiri
"""
import numpy as np
import pandas as pd
from datetime import timedelta

def resample_emg(fp, out_fp=None, time_col=0, start_col=1, end_col=3):
    '''
    

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
