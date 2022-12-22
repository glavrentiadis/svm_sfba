#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 07:49:32 2022

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
#user functions
sys.path.insert(0,'../python_lib/plotting')
import pylib_contour_plots as pycplt


#%% Define Variables
### ======================================
# Input/Output
# --------------------------------
#input flatfiles
fname_velprof_param = '../../Data/global_reg/bayesian_fit/JianFunUpd4dB_log_res_temp/all_trunc/all_trunc_stan_parameters.csv'
#geology info
fname_geol_info     = '../../Data/gis/vel_profiles_geology_info.csv'

#output directory
dir_out = '../../Data/misc/comparison_geology_dBr/'
dir_fig = dir_out + 'figures/'

#percentiles
prc2report = [16,84]

#%% Load Data
### ======================================
#load velocity profiles
df_velprof_param = pd.read_csv(fname_velprof_param)
#geology info
df_geolinfo      = pd.read_csv(fname_geol_info)

#%% Processing
### ======================================
#merge geology information with vel prof parameters
df_velprof_param = pd.merge(df_velprof_param, df_geolinfo[['DSID','VelID','Vs30_Mean','Geologic_U']], 
                            how='left', on=('DSID','VelID'))

#find unique geologic units
geol_u, geol_idx, geol_inv = np.unique(df_geolinfo.Geologic_U, return_index=True, return_inverse=True)
geol_ids = np.arange(len(geol_u)) + 1

#add geol id to df_geolinfo dataframe
df_velprof_param.loc[:,'Geologic_ID'] = geol_ids[geol_inv]

#create geologic class dataframe
df_geol_classes = pd.DataFrame({'Geologic_ID':geol_ids, 'Geologic_U':geol_u,
                                'Vs30_Mean':df_velprof_param.Vs30_Mean.values[geol_idx]})

#summarize vel param for different geological units
for j, g_u in enumerate(geol_u):
    #vel prof for given geologic unit
    df_vprof_p_geol = df_velprof_param.loc[df_velprof_param.Geologic_U==g_u,:]
    #extract between profiles values
    dBr_geol = df_vprof_p_geol.param_dBr_med
    #compute dBr statistics
    df_geol_classes.loc[j,'param_dBr_med']   = np.median(dBr_geol)
    df_geol_classes.loc[j,'param_dBr_mean']  = np.mean(dBr_geol)
    df_geol_classes.loc[j,'param_dBr_std']   = np.std(dBr_geol)
    for prc in prc2report:
        df_geol_classes.loc[j,r'param_dBr_%iprc'%prc] = np.percentile(dBr_geol,prc)


#%% Summary/Comparison
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#save files
fname_out = 'summary_geological_classes'
df_geol_classes.to_csv(dir_out + fname_out + '.csv', index=False)

#scatter of dBr
fname_fig = 'geol_dBr_stat_scatter'
fig, ax = plt.subplots(figsize = (20,10))
hl1 = ax.plot(df_velprof_param.Geologic_ID, df_velprof_param.param_dBr_med, 'o', markersize=6, color='black')
ax.plot([0, len(geol_u)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax.set_xlabel(r'Geologic Unit', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
# ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.set_xticks(df_geol_classes.Geologic_ID)
ax.set_xticklabels(df_geol_classes.Geologic_U)
ax.tick_params(axis='x', labelsize=25, rotation = 45)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([0, len(geol_u)+1])
ax.set_ylim([-8, +8])
ax.set_title(r'Geol Stat Scatter', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#stat summary of dBr
fname_fig = 'geol_dBr_stat_summary'
fig, ax = plt.subplots(figsize = (20,10))
hl1 = ax.plot(df_geol_classes.Geologic_ID, df_geol_classes.param_dBr_mean, 's', markersize=10, label='Mean', zorder=3)[0]
hl2 = ax.plot(df_geol_classes.Geologic_ID, df_geol_classes.param_dBr_med,  'o', markersize=8, label='Median')[0]
hl3 = ax.errorbar(df_geol_classes.Geologic_ID, df_geol_classes.param_dBr_med, 
                  yerr=np.abs(df_geol_classes[['param_dBr_16prc','param_dBr_84prc']].values-df_geol_classes.param_dBr_med.values[:,np.newaxis]).T, 
                  capsize=6, fmt='none', ecolor=hl2.get_color(), label='16/84th Percentile')
ax.plot([0, len(geol_u)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax.set_xlabel(r'Geologic Unit', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.set_xticks(df_geol_classes.Geologic_ID)
ax.set_xticklabels(df_geol_classes.Geologic_U)
ax.tick_params(axis='x', labelsize=25, rotation = 45)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([0, len(geol_u)+1])
ax.set_ylim([-2, +2])
ax.set_title(r'Geol Stat Summary', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


