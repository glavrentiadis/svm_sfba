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
import matplotlib.gridspec as gridspec
from matplotlib.ticker import  AutoLocator as plt_autotick
#user functions
sys.path.insert(0,'../python_lib/plotting')
import pylib_contour_plots as pycplt


#%% Define Variables
### ======================================
# Input/Output
# --------------------------------
#output filename
fname_out_main = 'all_trunc'

#input flatfiles
fname_velprof_param = '../../Data/regression/model_spatially_varying/'+fname_out_main+'_stan_parameters.csv'
#geology info
fname_geol_info     = '../../Data/gis/vel_profiles_geology_info.csv'

#output directory
dir_out = '../../Data/misc/comparison_geology_dBr/'
dir_fig = dir_out + 'figures/'

#percentiles
prc2report = [2,16,84,98]

#flat to plot titles
flag_titles = False

#geologic units
geol_units = {'af/Qi':                     ['af/Qi'],
              'Qi':                        ['Qi'],
              'Qal (flat)':                ['Qal1'],
              'Qal (moderate)':            ['Qal2'],
              'Qal (steep)':               ['Qal3'],
              'Qoa':                       ['Qoa'],
              'Qs':                        ['Qs'],
              'QT':                        ['QT'],
              'Tsh':                       ['Tsh'], 
              'Tss':                       ['Tss'], 
              'Tv':                        ['Tv'], 
              'Kss':                       ['Kss'],
              'KJf':                       ['KJf'],
              'serpentine':                ['sp'],
              'crystalline':               ['crystalline'],
              'water':                     ['water']}

#condensed geologic units
geol_units_grp = {'af/Qi':                     ['af/Qi'],
                  'Qi':                        ['Qi'],
                  'Qal':                       ['Qal1','Qal2','Qal3'],
                  'Qoa':                       ['Qoa'],
                  'Qs':                        ['Qs'],
                  'QT':                        ['QT'],
                  'Tsh, Tss, \& Tv':           ['Tsh','Tss','Tv'], 
                  'Kss \& KJf':                ['KJf','Kss'],
                  'crystalline \& serpentine': ['crystalline','sp'],
                  'water':                     ['water']}

#%% Load Data
### ======================================
#load velocity profiles
df_velprof_param = pd.read_csv(fname_velprof_param)
#geology info
df_geolinfo      = pd.read_csv(fname_geol_info)

#%% Processing
### ======================================
#merge geology information with vel prof parameters
df_velprof_param = pd.merge(df_velprof_param, df_geolinfo[['DSID','VelID','Wills15_Vs30_mean','Wills15_geologic_unit']], 
                            how='left', on=('DSID','VelID'))

#
df_velprof_param.loc[:,'Wills15_geologic_unit_ID']       = -1
df_velprof_param.loc[:,'Wills15_geologic_unit_Name']     = ''
df_velprof_param.loc[:,'Wills15_grp_geologic_unit_ID']   = -1
df_velprof_param.loc[:,'Wills15_grp_geologic_unit_Name'] = ''

#geological unit classification
for j, g_u in enumerate(geol_units):
    #identify geological units
    i_g = np.isin(df_velprof_param.Wills15_geologic_unit, geol_units[g_u])
    #classify geological units
    df_velprof_param.loc[i_g,'Wills15_geologic_unit_ID']   = j
    df_velprof_param.loc[i_g,'Wills15_geologic_unit_Name'] = g_u

#grouped geological unit classification
for j, g_u in enumerate(geol_units_grp):
    #identify geological units
    i_g = np.isin(df_velprof_param.Wills15_geologic_unit, geol_units_grp[g_u])
    #classify geological units
    df_velprof_param.loc[i_g,'Wills15_grp_geologic_unit_ID']   = j
    df_velprof_param.loc[i_g,'Wills15_grp_geologic_unit_Name'] = g_u

#create geologic class dataframe
df_geol_classes     = pd.DataFrame({'Wills15_geologic_unit_ID':range(len(geol_units)), 
                                    'Wills15_geologic_unit_Name':geol_units.keys()})
df_geol_grp_classes = pd.DataFrame({'Wills15_grp_geologic_unit_ID':range(len(geol_units_grp)), 
                                    'Wills15_grp_geologic_unit_Name':geol_units_grp.keys()})

#summarize vel param for different geological units
for j, g_u in df_geol_classes.iterrows():
    #vel prof for given geologic unit
    df_vprof_p_geol = df_velprof_param.loc[df_velprof_param.Wills15_geologic_unit_ID==g_u.Wills15_geologic_unit_ID,:]
    #extract between profiles values
    dBr_geol = df_vprof_p_geol.param_dBr_med
    #number of profiles
    df_geol_classes.loc[j,'nprof'] = len(dBr_geol)
    #compute dBr statistics
    df_geol_classes.loc[j,'param_dBr_med']   = np.median(dBr_geol)
    df_geol_classes.loc[j,'param_dBr_mean']  = np.mean(dBr_geol)
    df_geol_classes.loc[j,'param_dBr_std']   = np.std(dBr_geol)
    for prc in prc2report:
        df_geol_classes.loc[j,r'param_dBr_%iprc'%prc] = np.percentile(dBr_geol,prc)

#summarize vel param for different grouped geological units
for j, g_u in df_geol_grp_classes.iterrows():
    #vel prof for given geologic unit
    df_vprof_p_geol = df_velprof_param.loc[df_velprof_param.Wills15_grp_geologic_unit_ID==g_u.Wills15_grp_geologic_unit_ID,:]
    #extract between profiles values
    dBr_geol = df_vprof_p_geol.param_dBr_med
    #number of profiles
    df_geol_grp_classes.loc[j,'nprof'] = len(dBr_geol)
    #compute dBr statistics
    df_geol_grp_classes.loc[j,'param_dBr_med']   = np.median(dBr_geol)
    df_geol_grp_classes.loc[j,'param_dBr_mean']  = np.mean(dBr_geol)
    df_geol_grp_classes.loc[j,'param_dBr_std']   = np.std(dBr_geol)
    for prc in prc2report:
        df_geol_grp_classes.loc[j,r'param_dBr_%iprc'%prc] = np.percentile(dBr_geol,prc)

#%% Summary/Comparison
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#save files
fname_out = 'summary_geological_classes'
df_geol_classes.to_csv(dir_out + fname_out + '.csv', index=False)
fname_out = 'summary_grouped_geological_classes'
df_geol_grp_classes.to_csv(dir_out + fname_out + '.csv', index=False)

#original geological units
# ---   ---   ---   ---   ---   ---
#scatter of dBr
fname_fig = 'geol_dBr_stat_scatter'
fig, ax = plt.subplots(figsize = (20,10))
hl1 = ax.plot(df_velprof_param.Wills15_geologic_unit_ID, df_velprof_param.param_dBr_med, 'o', markersize=6, color='black')
ax.plot([-1, len(df_geol_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax.set_xlabel(r'Geologic Unit', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
# ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.set_xticks(df_geol_classes.Wills15_geologic_unit_ID)
ax.set_xticklabels(df_geol_classes.Wills15_geologic_unit_Name)
ax.tick_params(axis='x', labelsize=25, rotation = 45)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-1, len(df_geol_classes)])
ax.set_ylim([-2, +2])
if flag_titles: ax.set_title(r'Geological Units $\delta B_R$ Scatter', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#stat summary of dBr (16/84th percentile) 
fname_fig = 'geol_dBr_stat_summary1'
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(3, 1)
#plot axes
ax_main = plt.subplot(gs[1:3, 0])
ax_hist = plt.subplot(gs[0,   0], sharex=ax_main)
#
hl1 = ax_main.plot(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.param_dBr_mean, 's', markersize=10, label='Mean', zorder=3)[0]
hl2 = ax_main.plot(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.param_dBr_med,  'o', markersize=8, label='Median')[0]
hl3 = ax_main.errorbar(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.param_dBr_med, 
                       yerr=np.abs(df_geol_classes[['param_dBr_16prc','param_dBr_84prc']].values-df_geol_classes.param_dBr_med.values[:,np.newaxis]).T, 
                       capsize=6, fmt='none', ecolor=hl2.get_color(), label='16/84th Percentile')
ax_main.plot([-1, len(df_geol_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax_main.set_xlabel(r'Geologic Unit', fontsize=30)
ax_main.set_ylabel(r'$\delta B_r$',  fontsize=30)
ax_main.legend(loc='lower right', fontsize=30)
ax_main.grid(which='both')
ax_main.set_xticks(df_geol_classes.Wills15_geologic_unit_ID)
ax_main.set_xticklabels(df_geol_classes.Wills15_geologic_unit_Name)
ax_main.tick_params(axis='x', labelsize=25, rotation = 45)
ax_main.tick_params(axis='y', labelsize=25)
ax_main.set_xlim([-1, len(df_geol_classes)])
ax_main.set_ylim([-1.5, +1.5])
#number of profiles
hl_hist = ax_hist.bar(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.nprof.values, color="gray")
ax_hist.set_ylabel(f'Number of\nProfiles',  fontsize=30)
ax_hist.grid(which='both')
# ax_hist.tick_params(axis='x', labelsize=25)
ax_hist.tick_params(axis='y', labelsize=25)
# ax_hist.tick_params(axis='x', bottom = False)
ax_hist.tick_params(axis='x', which='both', labelbottom=False)
ax_hist.set_ylim(0,50)
#title
if flag_titles: ax_hist.set_title(r'Geological Units $\delta B_R$ Distribution', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#stat summary of dBr (2/98th percentile)
fname_fig = 'geol_dBr_stat_summary2'
fig = plt.figure(figsize=(20,10))
gs = gridspec.GridSpec(3, 1)
#plot axes
ax_main = plt.subplot(gs[1:3, 0])
ax_hist = plt.subplot(gs[0,   0], sharex=ax_main)
#
hl1 = ax_main.plot(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.param_dBr_mean, 's', markersize=10, label='Mean', zorder=3)[0]
hl2 = ax_main.plot(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.param_dBr_med,  'o', markersize=8, label='Median')[0]
hl3 = ax_main.errorbar(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.param_dBr_med, 
                       yerr=np.abs(df_geol_classes[['param_dBr_2prc','param_dBr_98prc']].values-df_geol_classes.param_dBr_med.values[:,np.newaxis]).T, 
                       capsize=6, fmt='none', ecolor=hl2.get_color(), label='2/98th Percentile')
ax_main.plot([-1, len(df_geol_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax_main.set_xlabel(r'Geologic Unit', fontsize=30)
ax_main.set_ylabel(r'$\delta B_r$',  fontsize=30)
ax_main.legend(loc='lower right', fontsize=30)
ax_main.grid(which='both')
ax_main.set_xticks(df_geol_classes.Wills15_geologic_unit_ID)
ax_main.set_xticklabels(df_geol_classes.Wills15_geologic_unit_Name)
ax_main.tick_params(axis='x', labelsize=25, rotation = 45)
ax_main.tick_params(axis='y', labelsize=25)
ax_main.set_xlim([-1, len(df_geol_classes)])
ax_main.set_ylim([-1.5, +1.5])
#number of profiles
hl_hist = ax_hist.bar(df_geol_classes.Wills15_geologic_unit_ID, df_geol_classes.nprof.values, color="gray")
ax_hist.set_ylabel(f'Number of\nProfiles',  fontsize=30)
ax_hist.grid(which='both')
# ax_hist.tick_params(axis='x', labelsize=25)
ax_hist.tick_params(axis='y', labelsize=25)
# ax_hist.tick_params(axis='x', bottom = False)
ax_hist.tick_params(axis='x', which='both', labelbottom=False)
ax_hist.set_ylim(0,50)
#title
if flag_titles: ax_hist.set_title(r'Geological Units $\delta B_R$ Distribution', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


#grouped geological units
# ---   ---   ---   ---   ---   ---
#scatter of dBr
fname_fig = 'geol_grp_dBr_stat_scatter'
fig, ax = plt.subplots(figsize = (20,10))
hl1 = ax.plot(df_velprof_param.Wills15_grp_geologic_unit_ID, df_velprof_param.param_dBr_med, 'o', markersize=6, color='black')
ax.plot([-1, len(df_geol_grp_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax.set_xlabel(r'Geologic Unit', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
# ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.set_xticks(df_geol_grp_classes.Wills15_grp_geologic_unit_ID)
ax.set_xticklabels(df_geol_grp_classes.Wills15_grp_geologic_unit_Name)
ax.tick_params(axis='x', labelsize=25, rotation = 90)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-1, len(df_geol_grp_classes)])
ax.set_ylim([-2, +2])
if flag_titles: ax.set_title(r'Geological Units $\delta B_R$ Scatter', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#stat summary of dBr (16/84th percentile) 
fname_fig = 'geol_grp_dBr_stat_summary1'
fig, ax = plt.subplots(figsize = (20,10))
hl1 = ax.plot(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_mean, 's', markersize=10, label='Mean', zorder=3)[0]
hl2 = ax.plot(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_med,  'o', markersize=8, label='Median')[0]
hl3 = ax.errorbar(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_med, 
                  yerr=np.abs(df_geol_grp_classes[['param_dBr_16prc','param_dBr_84prc']].values-df_geol_grp_classes.param_dBr_med.values[:,np.newaxis]).T, 
                  capsize=6, fmt='none', ecolor=hl2.get_color(), label='16/84th Percentile')
ax.plot([-1, len(df_geol_grp_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax.set_xlabel(r'Geologic Unit', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.set_xticks(df_geol_grp_classes.Wills15_grp_geologic_unit_ID)
ax.set_xticklabels(df_geol_grp_classes.Wills15_grp_geologic_unit_Name)
ax.tick_params(axis='x', labelsize=25, rotation = 90)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-1, len(df_geol_grp_classes)])
ax.set_ylim([-2, +2])
if flag_titles: ax.set_title(r'Geological Units $\delta B_R$ Distribution', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

# fig = plt.figure(figsize=(20,10))
# gs = gridspec.GridSpec(3, 1)
# #plot axes
# ax_main = plt.subplot(gs[1:3, 0])
# ax_hist = plt.subplot(gs[0,   0], sharex=ax_main)
# #
# hl1 = ax_main.plot(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_mean, 's', markersize=10, label='Mean', zorder=3)[0]
# hl2 = ax_main.plot(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_med,  'o', markersize=8, label='Median')[0]
# hl3 = ax_main.errorbar(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_med, 
#                        yerr=np.abs(df_geol_grp_classes[['param_dBr_16prc','param_dBr_84prc']].values-df_geol_grp_classes.param_dBr_med.values[:,np.newaxis]).T, 
#                        capsize=6, fmt='none', ecolor=hl2.get_color(), label='16/84th Percentile')
# ax_main.plot([-1, len(df_geol_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
# #edit properties
# ax_main.set_xlabel(r'Geologic Unit', fontsize=30)
# ax_main.set_ylabel(r'$\delta B_r$',  fontsize=30)
# ax_main.legend(loc='lower right', fontsize=30)
# ax_main.grid(which='both')
# ax_main.set_xticks(df_geol_grp_classes.Wills15_grp_geologic_unit_ID)
# ax_main.set_xticklabels(df_geol_grp_classes.Wills15_grp_geologic_unit_Name)
# ax_main.tick_params(axis='x', labelsize=25, rotation = 90)
# ax_main.tick_params(axis='y', labelsize=25)
# ax_main.set_xlim([-1, len(df_geol_grp_classes)])
# ax_main.set_ylim([-1.5, +1.5])
# #number of profiles
# hl_hist = ax_hist.bar(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.nprof.values, color="gray")
# ax_hist.set_ylabel(f'Number of\nProfiles',  fontsize=30)
# ax_hist.grid(which='both')
# # ax_hist.tick_params(axis='x', labelsize=25)
# ax_hist.tick_params(axis='y', labelsize=25)
# # ax_hist.tick_params(axis='x', bottom = False)
# ax_hist.tick_params(axis='x', which='both', labelbottom=False)
# ax_hist.set_ylim(0,50)
# #title
# if flag_titles: ax_hist.set_title(r'Geological Units $\delta B_R$ Distribution', fontsize=30)
# fig.tight_layout()
# fig.savefig( dir_fig + fname_fig + '.png' )

#stat summary of dBr (2/98th percentile)
fname_fig = 'geol_grp_dBr_stat_summary2'
fig, ax = plt.subplots(figsize = (20,10))
hl1 = ax.plot(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_mean, 's', markersize=10, label='Mean', zorder=3)[0]
hl2 = ax.plot(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_med,  'o', markersize=8, label='Median')[0]
hl3 = ax.errorbar(df_geol_grp_classes.Wills15_grp_geologic_unit_ID, df_geol_grp_classes.param_dBr_med, 
                  yerr=np.abs(df_geol_grp_classes[['param_dBr_2prc','param_dBr_98prc']].values-df_geol_grp_classes.param_dBr_med.values[:,np.newaxis]).T, 
                  capsize=6, fmt='none', ecolor=hl2.get_color(), label='2/98th Percentile')
ax.plot([-1, len(df_geol_grp_classes)+1],[0,0], color='black', linestyle='dashed', linewidth=3)
#edit properties
ax.set_xlabel(r'Geologic Unit', fontsize=30)
ax.set_ylabel(r'$\delta B_r$',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.set_xticks(df_geol_grp_classes.Wills15_grp_geologic_unit_ID)
ax.set_xticklabels(df_geol_grp_classes.Wills15_grp_geologic_unit_Name)
ax.tick_params(axis='x', labelsize=25, rotation = 90)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-1, len(df_geol_grp_classes)])
ax.set_ylim([-2, +2])
if flag_titles: ax.set_title(r'Geological Units $\delta B_R$ Distribution', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )
