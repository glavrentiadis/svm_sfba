#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:44:53 2024

@author: glavrent
"""

#load libraries
import os
import pathlib
import sys
#arithmetic libraries
import numpy  as np
import pandas as pd
from scipy import interpolate as interp
#ploting libraries
import matplotlib.pyplot as plt


#%% Define Variables
### ======================================
# Input/Output
# --------------------------------
#time history
fname_gm = '../../Data/site_reponse/ricker_ensemble_p10_10.txt'

#output directory
dir_out = '../../Data/site_reponse/'

#%% Load Data
### ======================================
#
gm_array = np.loadtxt(fname_gm)
#
dt = gm_array[1,0] - gm_array[0,0]



#%% Processing
### ======================================
#frequencies
freq = np.fft.fftfreq(gm_array.shape[0], dt)
#fourier amplitudes
fas  = np.abs(np.fft.fft(gm_array[:,1])) * dt

#keep positive frequencies
fas  = fas[freq>0]
freq = freq[freq>0]

#%% Plotting
### ======================================
#frequency content figure 
fig, ax = plt.subplots(figsize=(10,4))
fname_fig = 'ground_motion_time_domain'
#plot frequency content
hl = ax.plot(gm_array[:,0], gm_array[:,1], color='black')
#edit properties
ax.set_xlabel('Time (sec)', fontsize=32)
ax.set_ylabel('Acc. (g)',    fontsize=32)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.grid(which='both')
ax.set_xlim([0, 50])
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )

#frequency content figure 
fig, ax = plt.subplots(figsize=(10,10))
fname_fig = 'ground_motion_freq_content'
#plot frequency content
hl = ax.loglog(freq, fas, color='black')
#edit properties
ax.set_xlabel('Frequency (hz)', fontsize=32)
ax.set_ylabel('FAS (g sec)',    fontsize=32)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.grid(which='both')
ax.set_xlim([2e-2, 3e1])
ax.set_ylim([1e-5, 1e0])
#save figure
fig.tight_layout()
fig.savefig( dir_out + fname_fig + '.png' )

