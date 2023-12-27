#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:25:25 2023

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
sys.path.insert(0,'../python_lib/statistics')
sys.path.insert(0,'../python_lib/vel_prof')
from moving_mean import movingmean
from sigmoid import sigmoid
from calcvs0 import calcvs0

#%% Define Variables
### ======================================
#constants
z_star = 2.5

#Shi and Asimaki (2018) scaling coefficients
#Vs0 scaling
p1_as18 = -2.1688*10**(-4)
p2_as18 = 0.5182
p3_as18 = 69.452
#k scaling
r1_as18 =-59.67
r2_as18 =-0.2722
r3_as18 = 11.132
#n scaling
s1_as18 = 4.110
s2_as18 =-1.0521*10**(-4)
s3_as18 =-10.827
s4_as18 =-7.6187*10**(-3)

#Marafi et al. (2021) scaling coefficients
#Vs0 scaling
a0_m21 =-629
a1_m21 = 434
a2_m21 = 0.122
#n scaling
b0_m21 = 0.00912
b1_m21 = 0.646
b2_m21 =-0.201
b3_m21 = 0.136

#proposed model scaling coefficients
logVs30mid_new = 6.20947
logVs30scl_new = 0.372021 
r1_new         =-2.32772
r2_new         = 3.73403
r3_new         = 0.2734575
s2_new         = 4.83542

#vs30 array
vs30_array = np.logspace(np.log10(100), np.log10(2000), 1000)

#output directory
dir_fig = '../../Data/scaling_functions/'

#%% Postprocessing
### ======================================
#vs30 scaling
lnVs30s = (np.log(vs30_array)-logVs30mid_new)/logVs30scl_new

#compute z1.0
z1_array = np.exp( -7.15/4 * np.log( (vs30_array**4+571**4)/(1360**4+571**4) ) )

#proposed model
param_n_new     =         1      + s2_new * sigmoid(lnVs30s)
param_a_new     =-1/param_n_new
param_k_new     = np.exp( r1_new + r2_new * sigmoid(lnVs30s) + r3_new * logVs30scl_new * np.log(1 + np.exp(lnVs30s)) )
param_vs0_new   = np.array([calcvs0(vs30, k, n, z_star) for (vs30, n, k) in zip(vs30_array, param_n_new, param_k_new)])

#Shi and Asimaki (2018)
param_k_as18   = np.exp( r1_as18*(vs30_array)**r2_as18 + r3_as18 )
param_n_as18   = s1_as18*np.exp(s2_as18*vs30_array) + s3_as18*np.exp(s4_as18*vs30_array)
param_vs0_as18 = p1_as18*(vs30_array)**2 + p2_as18*vs30_array + p3_as18

#Marafi et al. (2021) 
param_vs0_m21 = a0_m21 + a1_m21*(vs30_array)**a2_m21
param_n_m21   = b0_m21 * (vs30_array)**b1_m21 * (z1_array)**b2_m21 * (vs30_array*z1_array)**b3_m21
param_k_m21   = ((1000-param_vs0_m21)/1000)**param_n_m21

#%% Plotting
### ======================================

#scaling k
fname_fig = 'scaling_comparison_k'
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.loglog(vs30_array, param_k_new,  '-', linewidth=4, zorder=10,  label='Proposed Model')
hl = ax.loglog(vs30_array, param_k_as18, '--', linewidth=4, zorder=10, label='Shi and Asimaki (2018)')
# hl = ax.loglog(vs30_array, param_k_m21,  '--', linewidth=4, zorder=10, label='Marafi et al. (2021)')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$k$',        fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_ylim([1e-2, 20])
# ax.set_title(r'$k$ Scaling', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig+fname_fig+'.png', bbox_inches='tight')

#scaling n
fname_fig = 'scaling_comparison_n'
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.semilogx(vs30_array, param_n_new,  '-',  linewidth=4, zorder=10, label='Proposed Model')
hl = ax.semilogx(vs30_array, param_n_as18, '--', linewidth=4, zorder=10, label='Shi and Asimaki (2018)')
hl = ax.semilogx(vs30_array, param_n_m21,  '--', linewidth=4, zorder=10, label='Marafi et al. (2021)')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$n$',       fontsize=30)
ax.legend(loc='upper left', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_yticks(np.arange(0,7,1))
ax.set_ylim([0,7])
# ax.set_title(r'$n$ Scaling', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig+fname_fig+'.png', bbox_inches='tight')

#scaling Vs0
fname_fig = 'scaling_comparison_vs0'
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.loglog([100,2000],[100,2000], linestyle=':', linewidth=3, color='black')
hl = ax.loglog(vs30_array, param_vs0_new,  '-',  linewidth=5, zorder=10,  label='Proposed Model')
hl = ax.loglog(vs30_array, param_vs0_as18, '--', linewidth=5, zorder=10, label='Shi and Asimaki (2018)')
hl = ax.loglog(vs30_array, param_vs0_m21,  '--', linewidth=5, zorder=10, label='Marafi et al. (2021)')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$V_{S0}$ (m/sec)',  fontsize=30)
ax.legend(loc='lower right', fontsize=30).set_zorder(10)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
# ax.set_ylim([10, 2000])
# ax.set_title(r'$V_{S0}$ Scaling', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig+fname_fig+'.png', bbox_inches='tight')
