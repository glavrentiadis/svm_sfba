#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 20:06:24 2025

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
#profile filename
fn_prof = '1-120_shi_asimaki_2018'

#number of spatially-varying realizations 
n_realiz_svar  = 10
#number of along-depth realizations
n_realiz_depth = 25

#column names for random realizations
cn_stat_realiz_dvar = ['Vs_drlz%i'%(l+1) for l in range(n_realiz_depth)]
cn_svar_realiz_mean = ['Vs_srlz%i'%(j+1) for j in range(n_realiz_svar)]
cn_svar_realiz_dvar = [['Vs_srlz%i_drlz%i'%(j+1,l+1) for j in range(n_realiz_svar)] for l in range(n_realiz_depth)]
cn_svar_realiz_dvar = sum(cn_svar_realiz_dvar,[])

#%% Load Data
### ======================================
#read profiles
df_prof_emp  = pd.read_csv(fn_prof + '_emp'  + '.csv')
df_prof_stat = pd.read_csv(fn_prof + '_stat' + '.csv')
df_prof_svar = pd.read_csv(fn_prof + '_svar' + '.csv')
df_prof_usgs = pd.read_csv(fn_prof + '_usgs' + '.csv')


#%% Plotting
### ======================================

#create figure - mean profiles
fig, ax = plt.subplots(figsize = (10,10))
#plot median profiles
hl_emp  = ax.step(df_prof_emp.Vs,  df_prof_emp.z,  linewidth=3.0, color='black', zorder=3, label='Velocity Profile')
hl_usgs = ax.plot(df_prof_usgs.Vs, df_prof_usgs.z, linewidth=3.0, color='black', zorder=2, linestyle='dashed', label=r'USGS SFBA Model')
hl_stat = ax.plot(df_prof_stat.Vs, df_prof_stat.z, linewidth=3.0, color='C1', zorder=1, label=f'Stationary Model')
hl_svar = ax.plot(df_prof_svar.Vs, df_prof_svar.z, linewidth=3.0, color='C0', zorder=1, label=f'Spatially Varying Model\n(Μean)')
#plot random realizations
hl_svar_realiz = ax.plot(df_prof_svar.loc[:,cn_svar_realiz_mean], df_prof_svar.z, linewidth=.5, color='C0', zorder=0, label=f'Spatially Varying Model\n(Uncertainty)')
#edit properties
ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
ax.set_ylabel('Depth (m)',      fontsize=30)
ax.grid(which='both')
ax.legend(handles=hl_emp+hl_usgs+hl_stat+hl_svar+[hl_svar_realiz[0]], loc='upper right', fontsize=30)
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.set_xlim([0, 2750])
ax.set_ylim([0, 150])
ax.invert_yaxis()
fig.tight_layout()
#save figure
fig.savefig( fn_prof + '_mean' + '.png' )

#create figure - stationary model
fig, ax = plt.subplots(figsize = (10,10))
#plot median profiles
hl_emp  = ax.step(df_prof_emp.Vs, df_prof_emp.z, linewidth=3.0, color='black', linestyle='solid', zorder=3, label='Velocity Profile')
hl_stat_mean = ax.plot(df_prof_stat.Vs, df_prof_stat.z, linewidth=3.0,  color='C1', linestyle='solid', zorder=1, label=f'Stationary Model\n(Mean)')
#plot along-depth random realizations
hl_stat_drlz = ax.plot(df_prof_stat.loc[:,cn_stat_realiz_dvar], df_prof_stat.z, linewidth=0.5, color='gray', linestyle='dashed', zorder=0, label=f'Stationary Model\n(Random Realizations)')
#edit properties
ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
ax.set_ylabel('Depth (m)',      fontsize=30)
ax.grid(which='both')
ax.legend(handles=hl_emp+hl_stat_mean+[hl_stat_drlz[0]], loc='lower left', fontsize=30)
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.set_xlim([0, 2750])
ax.set_ylim([0, 150])
ax.invert_yaxis()
fig.tight_layout()
#save figure
fig.savefig( fn_prof + '_stat' + '.png' )

#create figure - spatially model
fig, ax = plt.subplots(figsize = (10,10))
#plot median profiles
hl_emp  = ax.step(df_prof_emp.Vs, df_prof_emp.z, linewidth=3.0, color='black', linestyle='solid', zorder=3, label='Velocity Profile')
hl_svar_mean = ax.plot(df_prof_svar.Vs, df_prof_svar.z, linewidth=3.0,  color='C0', linestyle='solid', zorder=1, label=f'Spatially Varying Model\n(Mean)')
#plot along-depth random realizations
hl_svar_drlz = ax.plot(df_prof_svar.loc[:,cn_svar_realiz_dvar], df_prof_svar.z, linewidth=0.5, color='gray', linestyle='dashed', zorder=0, label=f'Spatially Varying Model\n(Random Realizations)')
#edit properties
ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
ax.set_ylabel('Depth (m)',      fontsize=30)
ax.grid(which='both')
ax.legend(handles=hl_emp+hl_svar_mean+[hl_svar_drlz[0]], loc='lower left', fontsize=30)
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
ax.set_xlim([0, 2750])
ax.set_ylim([0, 150])
ax.invert_yaxis()
fig.tight_layout()
#save figure
fig.savefig( fn_prof + '_svar' + '.png' )