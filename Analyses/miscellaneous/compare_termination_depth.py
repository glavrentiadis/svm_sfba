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
outlier_ds_id  = [3,  3,  3]
outlier_vel_id = [56, 57, 58]

#usgs vel model
dir_usgs   = '/mnt/halcloud_nfs/glavrent/Research/GP_Vel_profiles/Raw_files/vel_model/USGS_SFB_vel_model/'
fname_usgs = 'USGS_SFCVM_v21-1_detailed.h5'

flag_paper = True

#output directory
dir_out = '../../Data/misc/prof_termination/'
dir_fig = dir_out + 'figures/'

#%% Load Data
### ======================================
#load velocity profiles
#empirical
df_velprofs_data = pd.read_csv(fname_flatfile_data)

#remove columns with nan mid-depth
df_velprofs_data = df_velprofs_data.loc[~np.isnan(df_velprofs_data.Depth_MPt),:]

#truncate profiles at depth of 1000m/sec
if flag_trunc_z1:
    df_velprofs_data = df_velprofs_data.loc[~df_velprofs_data.flag_Z1,:]

#truncate profiles at depth of 1000m/sec
if flag_vs30_thres:
    df_velprofs_data = df_velprofs_data.loc[df_velprofs_data.Vs30>=Vs30_thres_min,:]



#%% Processing
### ======================================
#velocity profile information
vel_ids, vel_idx = np.unique(df_velprofs_data[['DSID','VelID']].values, axis=0, return_index=True)
n_vel = len(vel_idx)

#summary vel profile
df_vel_info =  df_velprofs_data[['DSID','DSName','VelID','VelName','Vs30','Lat','Lon']].iloc[vel_idx,:].reset_index(drop=True)

#initialize velocity model
model_vel_usgs = velmodel_usgs(dir_usgs, fname_usgs )

#iterate over all profiles
for k, v_ids in enumerate(vel_ids):
    print('process prof %i of %i (%s) ...'%(k+1,n_vel,df_vel_info.loc[k,'VelName']))
    #extract vel profile
    i_data = np.all(df_velprofs_data[['DSID','VelID']].values == v_ids, axis=1)
    
    #extract vel profile
    vs_data = df_velprofs_data.loc[i_data,'Vs']

    #query velocity model profile
    vs_usgs = model_vel_usgs.QueryZ(df_vel_info.loc[k,['Lat','Lon']].values, z=df_velprofs_data.loc[i_data,'Depth_MPt'].values)[2].Vs.values

    #residuals
    res     = np.log(vs_data) - np.log(vs_usgs)
    res_lin = vs_data - vs_usgs

    #store usgs vel and residuals
    df_velprofs_data.loc[i_data,'Vs_usgs']     = vs_usgs
    df_velprofs_data.loc[i_data,'res_usgs']    = res
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

# depth scaling
# ---------------------------
#displacement bins
d_bins = np.arange(0, 501, 100)

# vel profile / usgs residuals
# ---   ---   ---   ---   ---
#binned residuals
d_mbin, res_mmed, res_mmean, _, res_m16prc, res_m84prc = movingmean(df_velprofs_data.res_usgs, df_velprofs_data.Depth, d_bins)
#figure
fname_fig = ('residuals_usgs_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot([0,0],[0,500], color='k', linewidth=2)
hl1 = ax.plot(df_velprofs_data.res_usgs, df_velprofs_data.Depth, 'o',  markersize=4,  color='gray',  fillstyle='none')
hl2 = ax.plot(res_mmean,                 d_mbin,                 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(res_mmed,                  d_mbin,                 's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(res_mmed, d_mbin, xerr=np.abs(np.vstack((res_m16prc,res_m84prc)) - res_mmed ),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('residuals (log(V_{S~Obs})-log(V_{S~USGS}))', fontsize=30)
ax.set_ylabel('Depth (m)',                                  fontsize=30)
ax.legend(loc='lower left',                                 fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim([-2, 2])
ax.set_ylim([0, 500])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


# vel profile / usgs residuals (binned)
# ---   ---   ---   ---   ---
#residuals versus depth (Vs30 bins)
vs30_bins = [(0,100),(100,400),(400, 800),(800, 3000)]
for k, vs30_b in enumerate(vs30_bins):
    i_binned = np.logical_and(df_velprofs_data.Vs30 >= vs30_b[0],
                                             df_velprofs_data.Vs30 <  vs30_b[1])
    if i_binned.sum() == 0: continue
    df_velprofs_data_binned = df_velprofs_data.loc[i_binned,:].reset_index(drop=True)
    
    #binned residuals
    d_mbin, res_mmed, res_mmean, _, res_m16prc, res_m84prc = movingmean(df_velprofs_data_binned.res_usgs, df_velprofs_data_binned.Depth, d_bins)
    #figure
    fname_fig = ('residuals_usgs_versus_depth_binned_vs30_%i_%i'%vs30_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot([0,0],[0,500], color='k', linewidth=2)
    hl1 = ax.plot(df_velprofs_data_binned.res_usgs, df_velprofs_data_binned.Depth, 'o',  markersize=4,  color='gray',  fillstyle='none')
    hl2 = ax.plot(res_mmean,                 d_mbin,                               'd',  markersize=12, color='black', label='Mean')
    hl3 = ax.plot(res_mmed,                  d_mbin,                               's',  markersize=12, color='black', label='Median')
    hl4 = ax.errorbar(res_mmed, d_mbin, xerr=np.abs(np.vstack((res_m16prc,res_m84prc)) - res_mmed ),
                      capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                      label=r'$16-84^{th}$'+'\n Percentile')
    #edit properties
    ax.set_xlabel('residuals (log($V_{S~Obs}$)-log($V_{S~USGS}$))', fontsize=30)
    ax.set_ylabel('Depth (m)',                                  fontsize=30)
    ax.legend(loc='lower left',                                 fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlim([-2, 2])
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    if not flag_paper: ax.set_title(r'Residuals versus Depth', fontsize=30)
    fig.tight_layout()
    fig.savefig( dir_fig + fname_fig + '.png' )
    

# Vs scaling (USGS)
# ---------------------------
#vs bins
vs_bins = np.arange(100, 1500, 200)

# vel profile / usgs residuals
# ---   ---   ---   ---   ---
#binned residuals
vs_mbin, res_mmed, res_mmean, _, res_m16prc, res_m84prc = movingmean(df_velprofs_data.res_usgs, df_velprofs_data.Vs_usgs, vs_bins)
#figure
fname_fig = ('residuals_usgs_versus_vs-usgs').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot([0,0],[0,1500], color='k', linewidth=2)
hl1 = ax.plot(df_velprofs_data.res_usgs, df_velprofs_data.Vs_usgs, 'o',  markersize=4,  color='gray',  fillstyle='none')
hl2 = ax.plot(res_mmed,                  vs_mbin,                  'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(res_mmed,                  vs_mbin,                  's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(res_mmed, vs_mbin, xerr=np.abs(np.vstack((res_m16prc,res_m84prc)) - res_mmed ),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('residuals (log($V_{S~obs}$)-log($V_{S~USGS}$))', fontsize=30)
ax.set_ylabel('$V_{S~USGS}$ (m/sec)',                           fontsize=30)
ax.legend(loc='lower right',                                    fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim([-2, 2])
ax.set_ylim([0, 1400])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


# Vs scaling (Obs)
# ---------------------------
#vs bins
vs_bins = np.arange(100, 1500, 200)

# vel profile / usgs residuals
# ---   ---   ---   ---   ---
#binned residuals
vs_mbin, res_mmed, res_mmean, _, res_m16prc, res_m84prc = movingmean(df_velprofs_data.res_usgs, df_velprofs_data.Vs, vs_bins)
#figure
fname_fig = ('residuals_usgs_versus_vs-obs').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
ax.plot([0,0],[0,1500], color='k', linewidth=2)
hl1 = ax.plot(df_velprofs_data.res_usgs, df_velprofs_data.Vs,      'o',  markersize=4,  color='gray',  fillstyle='none')
hl2 = ax.plot(res_mmed,                  vs_mbin,                  'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(res_mmed,                  vs_mbin,                  's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(res_mmed, vs_mbin, xerr=np.abs(np.vstack((res_m16prc,res_m84prc)) - res_mmed ),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('residuals (log($V_{S~obs}$)-log($V_{S~USGS}$))', fontsize=30)
ax.set_ylabel('$V_{S~obs}$ (m/sec)',                           fontsize=30)
ax.legend(loc='lower right',                                    fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim([-2, 2])
ax.set_ylim([0, 1500])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )
