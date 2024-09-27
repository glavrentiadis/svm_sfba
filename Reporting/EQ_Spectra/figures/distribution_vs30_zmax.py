#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:29:10 2023

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
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import  AutoLocator as plt_autotick
from matplotlib.ticker import PercentFormatter as plt_prcntfrmt

#%% Load Data
### ======================================
#file name velocity profile info
fn_vprof_info = '../../../Data/vel_profiles_dataset/all_velocity_profles_info_valid.csv'
df_velprof_info = pd.read_csv(fn_vprof_info)

#
n_prof   = len(df_velprof_info)
wt_array = np.ones(n_prof)/n_prof

#%% Plot Data
### ======================================
fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(3, 3)
#plot axes
ax_main = plt.subplot(gs[1:3, :2])
ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)
#main plot
ax_main.scatter(df_velprof_info.Vs30.values, df_velprof_info.Z_max.values, marker='o',s=80, color="gray")
ax_main.set_xlabel('$V_{S30}$ (m/sec)',         fontsize=30)
ax_main.set_ylabel('Maximum Depth (m)',  fontsize=30)
ax_main.set_xscale('log')
ax_main.grid(which='both')
ax_main.tick_params(axis='x', labelsize=25)
ax_main.tick_params(axis='y', labelsize=25)
#maginal distribution (Vs30)
histVs30_prc, histVs30_bin, hl_histVs30 = ax_xDist.hist(df_velprof_info.Vs30.values,  bins=np.logspace(2,np.log10(2000), 10), 
                                                        align='mid', log=False, density=False, weights=wt_array, color="gray")
# ax_xDist.set_ylabel('Prob. Density\nFunction',  fontsize=30)
# ax_xDist.set_ylabel('Counts',  fontsize=30)
ax_xDist.set_ylabel('Percent (%)',  fontsize=30)
ax_xDist.grid(which='both')
ax_xDist.tick_params(axis='x', labelsize=25)
ax_xDist.tick_params(axis='y', labelsize=25)
#ax_xDist.set_ylim(0,0.003)
#ax_xDist.set_yticks([0,50,100])
#ax_xDist.set_ylim(0,125)
ax_xDist.set_ylim(0,1)
ax_xDist.yaxis.set_major_formatter(plt_prcntfrmt(1))
#ax_xDist.set_yticks([0,50,100])
#maginal distribution (Z_max)
histZmax_prc, histZmax_bin, hl_histZmax = ax_yDist.hist(df_velprof_info.Z_max.values, bins=np.arange(0,500.1,50),
                                                        align='mid', log=False, density=False, weights=wt_array, orientation='horizontal', color="gray")
#ax_yDist.set_xlabel('Prob. Density\nFunction',  fontsize=30)
#ax_yDist.set_xlabel('Counts',  fontsize=30)
ax_yDist.set_xlabel('Percent (%)',  fontsize=30)
ax_yDist.grid(which='both')
ax_yDist.tick_params(axis='x', labelsize=25)
ax_yDist.tick_params(axis='y', labelsize=0)
#ax_xDist.set_ylim(0,0.003)
#ax_yDist.set_xticks([0,50,100])
#ax_yDist.set_xlim(0,125)
ax_yDist.set_xlim(0,1)
ax_yDist.xaxis.set_major_formatter(plt_prcntfrmt(1))
fig.tight_layout()
fig.show()

fig.savefig('Vs30_z_max_distribution.png')
