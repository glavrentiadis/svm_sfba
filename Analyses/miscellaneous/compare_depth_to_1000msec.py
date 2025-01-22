#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 21:21:12 2025

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
import scipy
from scipy import interpolate as interp
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
#user functions
sys.path.insert(0,'../python_lib/statistics')
sys.path.insert(0,'../python_lib/usgs')
from moving_mean import movingmean
from query_usgs_v21 import USGSVelModelv21 as velmodel_usgs

#%% Define Variables
### ======================================
# Input/Output
# --------------------------------
fname_flatfile_data = '../../Data/vel_profiles_dataset/all_velocity_profles.csv'

#flag truncate
flag_trunc_z1 = False
# flag_trunc_z1 = True

#Vs30 minimum threshold
flag_vs30_thres = True
Vs30_thres_min = 100

#outlier profile ids
flag_outlier_rm = True
outlier_ds_id  = [1, 3]
outlier_vel_id = [6, 69]
# outlier_ds_id  = [1,  1,  3,  3,  3]
# outlier_vel_id = [57, 58, 56, 57, 58]

#usgs vel model
dir_usgs   = '/mnt/halcloud_nfs/glavrent/Research/GP_Vel_profiles/Raw_files/vel_model/USGS_SFB_vel_model/'
fname_usgs = 'USGS_SFCVM_v21-1_detailed.h5'

flag_paper = True

#output directory
dir_out = '../../Data/misc/z1_comparision/'
dir_fig = dir_out + 'figures/'

#%% Load Data
### ======================================
#load velocity profiles
df_vel_data = pd.read_csv(fname_flatfile_data)
#remove outliers
if flag_outlier_rm:
    #identify outliers
    i_out = np.any([np.logical_and(df_vel_data.DSID  == ds_id, df_vel_data.VelID == vel_id) 
                    for ds_id, vel_id in zip(outlier_ds_id, outlier_vel_id)], axis=0)
    #remove them and reset indices
    df_vel_data = df_vel_data.loc[~i_out,:].reset_index(drop=True)    



#%% Processing
### ======================================
#velocity profile information
vel_ids, vel_idx = np.unique(df_vel_data[['DSID','VelID']].values, axis=0, return_index=True)
n_vel = len(vel_idx)

#summary vel profile
df_vel_info =  df_vel_data[['DSID','DSName','VelID','VelName','Vs30','Lat','Lon']].iloc[vel_idx,:].reset_index(drop=True)
#
df_vel_info.loc[:,'Z1.0']     = np.nan
df_vel_info.loc[:,'Z1.0USGS'] = np.nan


#initialize velocity model
model_vel_usgs = velmodel_usgs(dir_usgs, fname_usgs )

#compute z1.0 in empirical profiles
print("Compute empirical z1:")
for k, v_ids in enumerate(vel_ids):
    print('\tprof %i of %i (%s) ...'%(k+1,n_vel,df_vel_info.loc[k,'VelName']))
    #extract vel profile
    i_data = np.all(df_vel_data[['DSID','VelID']].values == v_ids, axis=1)
    df_vel_prof = df_vel_data.loc[i_data,:]    

    #compute z1.0
    if df_vel_prof.Vs.max() < 1000.:
        z1 = np.nan
    else:
        i_z1 = max( np.argmin( np.abs(df_vel_prof.Vs - 1000.) ) - 1, 0 )
        z1 = df_vel_prof.Depth.values[i_z1]
       
    #store empirical value
    df_vel_info.loc[k,'Z1.0'] = z1
    
#compute z1.0 in usgs model
print("Compute USGS z1:")
for k, v_ids in enumerate(vel_ids):
    print('\tprof %i of %i (%s) ...'%(k+1,n_vel,df_vel_info.loc[k,'VelName']))
    
    #query z1.0
    z1usgs = model_vel_usgs.GetZ1_0(df_vel_info.loc[k,['Lat','Lon']].values)

    #store usgs value
    df_vel_info.loc[k,'Z1.0USGS'] = z1usgs * 1000.

#%% Save Data
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#save usgs residuals
df_vel_info.to_csv(dir_out + 'vel_prof_info_z1.0' + '.csv')

#%% Comparison
### ======================================

#z1 comparison
#figure
fname_fig = ('comparison_z1.0').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
#reference line
ax.loglog([1,1000],[1,1000], color='k', linewidth=2)
#plot z1 comparison
hl1 = ax.plot(df_vel_info['Z1.0'], df_vel_info['Z1.0USGS'], 'o',  markersize=10, color='black')
#edit properties
ax.set_xlabel('Measured Velocity Profiles: $Z_{1.0}$ (m)', fontsize=30)
ax.set_ylabel('USGS SFBA Velocity Model: $Z_{1.0}$ (m)',   fontsize=30)
# ax.legend(loc='lower left',                                     fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim([1,1000])
ax.set_ylim([1,1000])
if not flag_paper: ax.set_title(r'Comparison $Z_{1.0}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png', bbox_inches='tight')
