#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:47:22 2024

@author: glavrent
"""

#load libraries
import os
import sys
import pathlib
#string libraries
import re
#arithmetic libraries
import numpy as np
import numpy.matlib
from scipy.linalg import norm
from scipy.optimize import curve_fit
#statistics libraries
import pandas as pd
#semivariogram libraries
from skgstat import models
#plot libraries
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick

#user functions
def ExpKernel(loc_array, omega, ell, delta=0.001):
    
    #number of points
    n_pt = len(loc_array)
    #number of dimensions
    n_dim = loc_array.ndim

    #distance matrix
    dist_mat = np.array([norm(loc - loc_array, axis=1) if n_dim > 1 else np.abs(loc - loc_array)
                         for loc in loc_array])
    
    #covariance matrix
    cov_mat = omega**2 * np.exp(-dist_mat/ell)
    cov_mat += np.diag(np.full(n_pt, delta))
    
    #lower Cholesky decomposition
    chol_mat = np.linalg.cholesky(cov_mat)
    
    return cov_mat, chol_mat


#%% Define Variables
### ======================================
# Velocity model
#   1: Stationary
#   2: Spatially varying
flag_vel_model = 2

# Parameters
# --------------------------------
#define bin center points
z_max  = 50
n_bins = 10
z_bins = np.linspace(0, z_max, num=n_bins+1)
# z_bins = [0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]

#define bin limits
z_bins = [[z_bins[j], z_bins[j+1]] for j in range(len(z_bins)-1)]

flag_sill_fixed = False

#column name
if   flag_vel_model == 1:
    col_res = 'res_tot'
elif flag_vel_model == 2:
    col_res = 'res_dW'

# Input/Output
# --------------------------------
#model name
if   flag_vel_model == 1:
    fname_model = 'model_stationary'
elif flag_vel_model == 2:
    fname_model = 'model_spatially_varying'     

#input flatfile
fname_flatfile = f'../../Data/regression/{fname_model}/all_trunc_stan_residuals.csv'
fname_hypparm  = f'../../Data/regression/{fname_model}/all_trunc_stan_hyperparameters.csv'    

#output directory
dir_out = '../../Data/regression/along_depth_correlation/'
dir_fig = dir_out + 'figures/'
#flag for paper quality paper
flag_paper = True

#%% Load Data
### ======================================
#load velocity profiles
df_res = pd.read_csv(fname_flatfile)
#load hyper-parameters
df_hyp = pd.read_csv(fname_hypparm, index_col=0)


#%% Regression
### ======================================
#bin width and cencer
z_bins_center = np.array([np.mean(z_b) for z_b in z_bins])
z_bins_width  = np.array([np.diff(z_b)[0] for z_b in z_bins])

#identify unique profiles
vel_id_dsid = np.unique(df_res[['DSID','VelID']].values, axis=0)
n_vel = vel_id_dsid.shape[0]

#initialize bin
binned_res  = [[] for k in range(len(z_bins))]
binned_z = [[] for k in range(len(z_bins))]

#compute binned residuals
for j in range(vel_id_dsid.shape[0]):
    print('Bbnning residuals: vel. profile %i of %i ...'%(j+1, n_vel))

    #select profile vel measurements
    df_vel =  df_res.loc[np.all(df_res.loc[:,['DSID','VelID']] == vel_id_dsid[j,:],axis=1),:].reset_index(drop=True)
    
    #compute distance matrix
    dist_mat  = np.asarray( [np.abs(df_vel.Depth_MPt.values-z) for z in df_vel.Depth_MPt.values] )
    dist_mat += np.tril(np.full(dist_mat.shape, np.nan),1)

    #identify residuals within each bin
    i_pair_binned = [np.argwhere(np.logical_and(dist_mat>=z_b[0], dist_mat<z_b[1])) for z_b in z_bins]
    
    #binned residuals
    for k in range(len(z_bins)):
        binned_res[k] += [df_vel.loc[i_p,col_res].tolist()    for i_p in i_pair_binned[k]]
        binned_z[k]   += [dist_mat[i_p[0],i_p[1]].tolist() for i_p in i_pair_binned[k]]

#convert to numpy array
binned_res  = [np.asarray(r_b) for r_b in binned_res]
binned_z    = [np.asarray(d_z) for d_z in binned_z]
binned_npt  = np.asarray([len(d_b) for d_b in binned_z])

#comupte semivariogram
emp_svar_array = [np.mean(np.diff(b_r,axis=1)**2)/2 for b_r in binned_res]
emp_z_array    = [np.mean(b_z)                      for b_z in binned_z]

#define semi-variance model
#svarmodel =  models.spherical if not flag_sill_fixed else lambda h, l:  models.spherical(h, l, c0=df_hyp.loc['mean','sigma_vel']**2)   
svarmodel =  models.exponential if not flag_sill_fixed else lambda h, l:  models.exponential(h, l, c0=df_hyp.loc['mean','sigma_vel']**2)   
    
#fit covariance model
seed0 = [np.mean(emp_z_array)] if flag_sill_fixed else [np.mean(emp_z_array), np.mean(emp_svar_array)]
svar_cof_mu, svar_cof_var = curve_fit(svarmodel, emp_z_array, emp_svar_array, p0=seed0, sigma=1/np.sqrt(binned_npt))
svar_cof_sd = np.sqrt(np.diag(svar_cof_var))

#evaluate model
model_z_array    = np.linspace(0,np.max(z_bins),200)
model_svar_array = svarmodel(model_z_array, svar_cof_mu[0]) if flag_sill_fixed else svarmodel(model_z_array, svar_cof_mu[0], svar_cof_mu[1])

#summarize coefficinets
df_svar_coeffs = pd.DataFrame(np.vstack([svar_cof_mu,svar_cof_sd]), index=['mean','std_err'], columns=('range','sill'))

#%% Output
### ======================================
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#summarize coefficinets
df_svar_coeffs.to_csv(dir_out + (fname_model+'_semi_variogram_parameters').replace(' ','_') + '.csv')

#create semi-variogram and bin size figure
fname_fig = (fname_model+'_semi_variogram').replace(' ','_')
fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(3, 1)
#plot axes
ax_main = plt.subplot(gs[1:,0])
ax_bins = plt.subplot(gs[0,0],sharex=ax_main)
#main plot
ax_main.scatter(emp_z_array, emp_svar_array, marker='o',s=80, color="gray")
ax_main.plot(model_z_array, model_svar_array, linewidth=2, color='k')
ax_main.set_xlabel('Vertical separation distance (m)',  fontsize=30)
ax_main.set_ylabel('Semi-variance',  fontsize=30)
ax_main.grid(which='both')
ax_main.tick_params(axis='x', labelsize=28)
ax_main.tick_params(axis='y', labelsize=28)
ax_main.set_xlim([0, np.max(z_bins)]) 
ax_main.set_ylim([0, .15]) 
#bin size
ax_bins.bar(z_bins_center, binned_npt, z_bins_width*.9, color="gray")
ax_bins.set_ylabel('Bin size',  fontsize=30)
ax_bins.grid(which='both')
ax_bins.tick_params(axis='x', labelsize=0)
ax_bins.tick_params(axis='y', labelsize=28)
# ax_bins.set_ylim([0, 100000]) 
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#create semi-variogram figure (only)
fname_fig = (fname_model+'_semi_variogram2').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,5))
#main plot
ax.scatter(emp_z_array, emp_svar_array, marker='o',s=80, color="gray")
ax.plot(model_z_array, model_svar_array, linewidth=2, color='k')
ax.set_xlabel('Vertical separation distance (m)',  fontsize=30)
ax.set_ylabel('Semi-variance',  fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim([0, np.max(z_bins)]) 
ax.set_ylim([0, .15]) 
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')
