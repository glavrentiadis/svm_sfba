#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:46:04 2022

@author: glavrent
"""

#load variables
import os
import sys
import pathlib
#string libraries
import re
#arithmetic libraries
import numpy as np
import numpy.matlib
from scipy import interpolate as interp
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick

#%% Define Variables
### ======================================
# Input/Output
# --------------------------------
#input flatfile
fname_velprof_info = '../../Data/gis/vel_profiles_geology_info.csv'

#output directory
dir_out = '../../Data/misc/comparison_Wills15_Vs30/'
dir_fig = dir_out + 'figures/'


#%% Load Data
### ======================================
#load velocity profiles
df_velprof_info = pd.read_csv(fname_velprof_info)


#%% Comparison
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

# Comparison Vs30
# ---------------------------
fname_fig = 'comparison_Wills15_Vs30'
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot([0,1500], [0,1500], color='black', linewidth=2)
hl = ax.plot(df_velprof_info.Vs30, df_velprof_info.Vs30_Mean, 'o',markersize=8)
#edit properties
ax.set_xlabel('Computed $V_{S30}$',   fontsize=30)
ax.set_ylabel('Wills 2015 $V_{S30}$', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([0, 1250])
ax.set_ylim([0, 1250])
ax.set_title(r'Comparison Vs30 Wills 2015', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Correlation Vs30
# ---------------------------
rho = np.corrcoef(df_velprof_info.Vs30.values, df_velprof_info.Vs30_Mean.values)
print('Correlation coefficient: %.2f'%rho[0,1])
