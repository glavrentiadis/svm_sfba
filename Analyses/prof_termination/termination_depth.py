#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:03:56 2023

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
from moving_mean import movingmean

#%% Define Variables
### ======================================
# Input/Output
# --------------------------------
fname_flatfile_data = '../../Data/vel_profiles_dataset/all_velocity_profles.csv'
fname_flatfile_usgs = '../../Data/vel_profiles_usgs/all_velocity_profles.csv'

#output directory
dir_out = '../../Data/prof_termination/'
dir_fig = dir_out + 'figures/'


#%% Load Data
### ======================================
#load velocity profiles
#empirical
df_velprofs_data = pd.read_csv(fname_flatfile_data)
#usgs
df_velprofs_usgs = pd.read_csv(fname_flatfile_usgs)

#remove rows with nan mid-depth
df_velprofs_data = df_velprofs_data.loc[~np.isnan(df_velprofs_data.Depth_MPt),:]
df_velprofs_usgs = df_velprofs_usgs.loc[~np.isnan(df_velprofs_usgs.Depth_MPt),:]


#%% Processing
### ======================================
#velocity profile information
vel_ids, vel_idx = np.unique(df_velprofs_data[['DSID','VelID']].values, axis=0, return_index=True)
n_vel = len(vel_idx)

#summary vel profile
df_vel_info =  df_velprofs_data[['DSID','DSName','VelID','VelName','Vs30','Lat','Lon']].iloc[vel_idx,:].reset_index(drop=True)

#iterate over all profiles
for k, v_ids in enumerate(vel_ids):
    #extract vel profile
    i_data = np.all(df_velprofs_data[['DSID','VelID']].values == v_ids, axis=1)
    i_usgs = np.all(df_velprofs_usgs[['DSID','VelID']].values == v_ids, axis=1)
    df_v_data = df_velprofs_data.loc[i_data,:]
    df_v_usgs = df_velprofs_usgs.loc[i_usgs,:]

    #progress report
    if i_usgs.sum():
        print('process prof %i of %i (%s) ...'%(k+1,n_vel,df_vel_info.loc[k,'VelName']))
    else:
        print('\t skip prof %i of %i (%s) ...'%(k+1,n_vel,df_vel_info.loc[k,'VelName']))
        continue
    
    #interpolate vel profile
    vs_usgs = interp.interp1d(x=df_v_usgs.Depth,y=df_v_usgs.Vs, kind='next', bounds_error=False, 
                              fill_value=tuple(df_v_usgs.Vs.values[[0,-1]]))(df_v_data.Depth)

    #comptue residuals
    res     = np.log(df_v_data.Vs) - np.log(vs_usgs)
    res_lin = df_v_data.Vs - vs_usgs
    
    #store usgs vel and residuals
    df_velprofs_data.loc[i_data,'Vs_usgs']  = vs_usgs
    df_velprofs_data.loc[i_data,'res_usgs'] = res
    df_velprofs_data.loc[i_data,'reslin_usgs'] = res_lin

#remove rows with nan residuals
df_velprofs_data = df_velprofs_data.loc[~np.isnan(df_velprofs_data.res_usgs),:]

#%% Save Data
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#save usgs residuals
df_velprofs_data.to_csv(dir_out + 'usgs_residuals' + '.csv')

#%% Comparison
### ======================================
#displacement bins
d_bins = np.arange(0, 501, 100)
#binned residuals
d_mbin, res_mmed, _, _, res_m16prc, res_m84prc = MovingMean(df_velprofs_data.res_usgs, df_velprofs_data.Depth, d_bins)

# vel profile / usgs residuals
# ---------------------------
fname_fig = ('residuals_usgs_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot([0,0],[0,500], color='k', linewidth=2)
hl1 = ax.plot(df_velprofs_data.res_usgs, df_velprofs_data.Depth, 'o',  markersize=2, label='Residuals')
hl2 = ax.plot(res_mmed,                  d_mbin,                 'o',  markersize=6, label='Median')
hl3 = ax.errorbar(res_mmed, d_mbin, xerr=np.abs(np.vstack((res_m16prc,res_m84prc)) - res_mmed ),
                  capsize=4, fmt='none', ecolor=hl2[0].get_color(), label='16/84 Percentile')
#edit properties
ax.set_xlabel('residuals (log(Vel)-log(USGS))', fontsize=30)
ax.set_ylabel('Depth (m)',                      fontsize=30)
ax.legend(loc='lower left',                     fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 500])
ax.invert_yaxis()
ax.set_title(r'Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# Plot Vel Profiles
# ---------------------------
#iterate vel profiles
for k, v_ids in enumerate(vel_ids):
    print('Plotting vel. profile and residuals  %i of %i'%(k+1, n_vel))
    #extract vel profile of interest
    i_data = np.all(df_velprofs_data[['DSID','VelID']].values == v_ids, axis=1)
    i_usgs = np.all(df_velprofs_usgs[['DSID','VelID']].values == v_ids, axis=1)
    if i_data.sum() == 0: continue #skip profile if unavailable
    df_v_data = df_velprofs_data.loc[i_data,:]
    df_v_usgs = df_velprofs_usgs.loc[i_usgs,:]
    
    #profile name
    name_vel =  (df_v_data.DSName.values[0]+' '+df_v_data.VelName.values[0]).replace('_',' ')
    fname_vel = ('prof_' + name_vel + '_vel').replace(' ','_')

    #velocity profile
    depth_array_data   = np.concatenate([[0], np.cumsum(df_v_data.Thk)])
    vel_array_data     = np.concatenate([df_v_data.Vs.values,      [df_v_data.Vs.values[-1]]])
    
    #create figure   
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.step(vel_array_data,       depth_array_data,  label='Velocity Profile')
    hl2 = ax.step(df_v_usgs.Vs.values, df_v_usgs.Depth,    label='USGS Profile')
    hl3 = ax.step(df_v_data.Vs_usgs,   df_v_data.Depth, 'o', color=hl2[0].get_color(),
                  label='USGS Interpolation')
    #edit properties
    ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.legend(loc='lower left', fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.set_xlim([0, 2000])
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    ax.set_title(name_vel, fontsize=30)
    fig.tight_layout()
    fig.savefig( dir_fig + fname_vel + '.png' )
 
# Plot Vel Residuals
# ---------------------------
#iterate vel profiles
for k, v_ids in enumerate(vel_ids):
    print('Plotting vel. profile and residuals  %i of %i'%(k+1, n_vel))
    #extract vel profile of interest
    i_data = np.all(df_velprofs_data[['DSID','VelID']].values == v_ids, axis=1)
    if i_data.sum() == 0: continue #skip profile if unavailable
    df_v_data = df_velprofs_data.loc[i_data,:]
    
    #profile name
    name_vel =  (df_v_data.DSName.values[0]+' '+df_v_data.VelName.values[0]).replace('_',' ')
    fname_vel = ('prof_' + name_vel + '_res').replace(' ','_')

    #create figure   
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot([0,0],[0,500], color='k', linewidth=2)
    hl1 = ax.step(df_v_data.res_usgs, df_v_data.Depth, 'o')
    #edit properties
    ax.set_xlabel('residuals (log(Vel)-log(USGS))', fontsize=30)
    ax.set_ylabel('Depth (m)',                      fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.set_xlim([-3, 3])
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    ax.set_title(name_vel, fontsize=30)
    fig.tight_layout()
    fig.savefig(dir_fig + fname_vel + '.png' )
