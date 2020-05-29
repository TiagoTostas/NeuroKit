#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
#sys.path.append('/Users/tiagorodrigues/Documents/GitHub/NeuroKit')
sys.path.append(r"C:\Users\Tiago Rodrigues\Documents\GitHub\Neurokit")

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import neurokit2 as nk

from pathlib import Path

# ==== Helper functions ====


def open_file(typeactivity,dayactivity, show):
    # Input: data file name
    # e.g  "Activities/20200510-TR-Belt.txt"
    
    # add path of the data
    #sys.path.append('/Users/tiagorodrigues/OneDrive - Universidade de Lisboa/Ana Luisa Nobre Fred - TiagoRodrigues_EPFL_FieldWiz_tese_2020/Data/')
    sys.path.append(r"C:\Users\Tiago Rodrigues\Universidade de Lisboa\Ana Luisa Nobre Fred - TiagoRodrigues_EPFL_FieldWiz_tese_2020\Data")
    
    # name of the file
    data_folder = Path(r"C:\Users\Tiago Rodrigues\Universidade de Lisboa\Ana Luisa Nobre Fred - TiagoRodrigues_EPFL_FieldWiz_tese_2020\Data")   
    file_to_open = os.path.join(data_folder, typeactivity,dayactivity)

    # import header json file
    f = open(file_to_open) 
    line = f.readline()
    line = f.readline(1)
    line = f.readline()
    header = line.replace("\n", "")
    
    # import ecg data
    ecg_fieldwiz = np.genfromtxt(file_to_open) 
    time = np.arange(0, len(ecg_fieldwiz)/250, 1/250)
    
    
    # plot ECG
    if show == 1:
        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(time,ecg_fieldwiz)
        plt.ylabel("ECG (16-bit)")
        plt.xlabel("Time (s)")
        plt.title('Raw ECG', fontdict=None, loc='center', pad=None)
    
    return header,ecg_fieldwiz,time
    
    
    
def rr_artefacts(rr, c1=0.13, c2=0.17, alpha=5.2):
    """Artefacts detection from RR time series using the subspaces approach
    proposed by Lipponen & Tarvainen (2019).
    Parameters
    ----------
    rr : 1d array-like
        Array of RR intervals.
    c1 : float
        Fixed variable controling the slope of the threshold lines. Default is
        0.13.
    c2 : float
        Fixed variable controling the intersect of the threshold lines. Default
        is 0.17.
    alpha : float
        Scaling factor used to normalize the RR intervals first deviation.
    Returns
    -------
    artefacts : dictionnary
        Dictionnary storing the parameters of RR artefacts rejection. All the
        vectors outputed have the same length than the provided RR time serie:
        * subspace1 : 1d array-like
            The first dimension. First derivative of R-R interval time serie.
        * subspace2 : 1d array-like
            The second dimension (1st plot).
        * subspace3 : 1d array-like
            The third dimension (2nd plot).
        * mRR : 1d array-like
            The mRR time serie.
        * ectopic : 1d array-like
            Boolean array indexing probable ectopic beats.
        * long : 1d array-like
            Boolean array indexing long RR intervals.
        * short : 1d array-like
            Boolean array indexing short RR intervals.
        * missed : 1d array-like
            Boolean array indexing missed RR intervals.
        * extra : 1d array-like
            Boolean array indexing extra RR intervals.
        * threshold1 : 1d array-like
            Threshold 1.
        * threshold2 : 1d array-like
            Threshold 2.
    Notes
    -----
    This function will use the method proposed by Lipponen & Tarvainen [1]_ to
    detect ectopic beats, long, shorts, missed and extra RR intervals.
    References
    ----------
    [1] Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for
        heart rate variability time series artefact correction using novel
        beat classification. Journal of Medical Engineering & Technology,
        43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306
    """
    if isinstance(rr, list):
        rr = np.array(rr)

    ###########
    # Detection
    ###########

    # Subspace 1 (dRRs time serie)
    dRR = np.diff(rr, prepend=0)
    dRR[0] = dRR[1:].mean()  # Set first item to a realistic value

    dRR_df = pd.DataFrame({'signal': np.abs(dRR)})
    q1 = dRR_df.rolling(
        91, center=True, min_periods=1).quantile(.25).signal.to_numpy()
    q3 = dRR_df.rolling(
        91, center=True, min_periods=1).quantile(.75).signal.to_numpy()

    th1 = alpha * ((q3 - q1) / 2)
    dRR = dRR / th1
    s11 = dRR

    # mRRs time serie
    medRR = pd.DataFrame({'signal': rr}).rolling(
                    11, center=True, min_periods=1).median().signal.to_numpy()
    mRR = rr - medRR
    mRR[mRR < 0] = 2 * mRR[mRR < 0]

    mRR_df = pd.DataFrame({'signal': np.abs(mRR)})
    q1 = mRR_df.rolling(
        91, center=True, min_periods=1).quantile(.25).signal.to_numpy()
    q3 = mRR_df.rolling(
        91, center=True, min_periods=1).quantile(.75).signal.to_numpy()

    th2 = alpha * ((q3 - q1) / 2)
    mRR /= th2

    # Subspace 2
    ma = np.hstack(
        [0, [np.max([dRR[i-1], dRR[i+1]]) for i in range(1, len(dRR)-1)], 0])
    mi = np.hstack(
        [0, [np.min([dRR[i-1], dRR[i+1]]) for i in range(1, len(dRR)-1)], 0])
    s12 = ma
    s12[dRR < 0] = mi[dRR < 0]

    # Subspace 3
    ma = np.hstack(
        [[np.max([dRR[i+1], dRR[i+2]]) for i in range(0, len(dRR)-2)], 0, 0])
    mi = np.hstack(
        [[np.min([dRR[i+1], dRR[i+2]]) for i in range(0, len(dRR)-2)], 0, 0])
    s22 = ma
    s22[dRR >= 0] = mi[dRR >= 0]

    ##########
    # Decision
    ##########

    # Find ectobeats
    cond1 = (s11 > 1) & (s12 < (-c1 * s11-c2))
    cond2 = (s11 < -1) & (s12 > (-c1 * s11+c2))
    ectopic = cond1 | cond2
    # No ectopic detection and correction at time serie edges
    ectopic[-2:] = False
    ectopic[:2] = False

    # Find long or shorts
    longBeats = \
        ((s11 > 1) & (s22 < -1)) | ((np.abs(mRR) > 3) & (rr > np.median(rr)))
    shortBeats = \
        ((s11 < -1) & (s22 > 1)) | ((np.abs(mRR) > 3) & (rr <= np.median(rr)))

    # Test if next interval is also outlier
    for cond in [longBeats, shortBeats]:
        for i in range(len(cond)-2):
            if cond[i] is True:
                if np.abs(s11[i+1]) < np.abs(s11[i+2]):
                    cond[i+1] = True

    # Ectopic beats are not considered as short or long
    shortBeats[ectopic] = False
    longBeats[ectopic] = False

    # Missed vector
    missed = np.abs((rr/2) - medRR) < th2
    missed = missed & longBeats
    longBeats[missed] = False  # Missed beats are not considered as long

    # Etra vector
    extra = np.abs(rr + np.append(rr[1:], 0) - medRR) < th2
    extra = extra & shortBeats
    shortBeats[extra] = False  # Extra beats are not considered as short

    # No short or long intervals at time serie edges
    shortBeats[0], shortBeats[-1] = False, False
    longBeats[0], longBeats[-1] = False, False

    artefacts = {'subspace1': s11, 'subspace2': s12, 'subspace3': s22,
                 'mRR': mRR, 'ectopic': ectopic, 'long': longBeats,
                 'short': shortBeats, 'missed': missed, 'extra': extra,
                 'threshold1': th1, 'threshold2': th2}
    

    return artefacts
    
