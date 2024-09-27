#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:41:17 2024

@author: glavrent
"""

#load libraries
import os
import re
import pathlib
import sys
#arithmetic libraries
import numpy  as np
import pandas as pd
from scipy import interpolate as interp
#ploting libraries
import matplotlib.pyplot as plt
#user functions
sys.path.insert(0,'../../../Analyses/python_lib/statistics')
sys.path.insert(0,'../../../Analyses/python_lib/vel_prof')
sys.path.insert(0,'../../../Analyses/python_lib/plotting')
import pylib_contour_plots as pycplt

#%% Define Variables
### ======================================
#file name spatially varying coefficient
fn_svar_params = '../../../Data/regression/model_spatially_varying/all_trunc_stan_parameters.csv'

#output filename
fname_out_main = 'svarying_model'

#
dir_fig = ''

#%% Load Data
### ======================================
#
df_svar_params = pd.read_csv(fn_svar_params)


#%% Plotting
### ======================================
#


# Spatial variability
# ---------------------------
# pl_win = [[36, 39], [-124, -121]]
pl_win = [[36.5, 38.75], [-123.25, -121.25]]

#median delta B r
data2plot = df_svar_params[['Lat','Lon','param_dBr_med']].values

fname_fig =  (fname_out_main + '_locations_param_dBr_med').replace(' ','_')
fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-0.5,  cmax=0.5, flag_grid=False, 
                                                      title=None, cbar_label='', log_cbar = False, 
                                                      frmt_clb = '%.2f', alpha_v = 0.7, cmap='viridis', marker_size=70.)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'Median: $\delta B_r$', size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#std delta B r
data2plot = df_svar_params[['Lat','Lon','param_dBr_std']].values

fname_fig =  (fname_out_main + '_locations_param_dBr_std').replace(' ','_')
fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=0,  cmax=.50, flag_grid=False, 
                                                      title=None, cbar_label='', log_cbar = False, 
                                                      frmt_clb = '%.2f', alpha_v = 0.7, cmap='Wistia', marker_size=70.)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'Standard Deviation: $\delta B_r$', size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
#grid lines
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}
#save figure
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )
