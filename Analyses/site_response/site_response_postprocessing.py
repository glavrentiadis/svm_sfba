#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:50:01 2024

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


#%% Define Variables
### ======================================

# Scorring
# ---   ---   ---   ---   ---
#directory version
ver = 3
#input ground motion type
gm_type = 'outcrop'
# gm_type = 'incident'
#score type
# score_type = 'S5_RMS_Acceleration'
# score_type = 'S6_RMS_Velocity'
# score_type = 'S7_RMS_Displacement'
# score_type = 'S8_Spectral_Acceleration'
# score_type = 'S9_Fourier_Spectra'
score_type = 'S10_Average_Score'


#directory
dir_score = '../../Data/site_reponse/gof/ver%i/%s/'%(ver, gm_type)

#filenames (binned)
if ver == 1:
    fn_score_binned = {'f_0.2-2.0hz':  '%s_fmin0.2_fmax2.csv'%(score_type),
                       'f_2.0-5.0hz':  '%s_fmin2_fmax5.csv'%(score_type),
                       'f_5.0-15.0hz': '%s_fmin5_fmax15.csv'%(score_type)}
elif ver == 2:
    fn_score_binned = {'f_0.2-1.0hz':  '%s_fmin0.2_fmax1.0.csv'%(score_type),
                       'f_1.0-2.5hz':  '%s_fmin1.0_fmax2.5.csv'%(score_type),
                       'f_2.5-5.0hz': '%s_fmin2.5_fmax5.0.csv'%(score_type),
                       'f_5.0-10.0hz': '%s_fmin5.0_fmax10.0.csv'%(score_type)}
elif ver == 3:
    fn_score_binned = {'0.01-fp':  '%s_fmin0.01_fmaxfp.csv'%(score_type),
                       'fp-2fp':  '%s_fminfp_fmax2fp.csv'%(score_type),
                       '2fp-10hz': '%s_fmin2fp_fmax10.0.csv'%(score_type)}

#filenames (total)
if ver == 1:
    fn_score_total = '%s_fmin0.2_fmax15.csv'%(score_type)
elif ver == 2:
    fn_score_total = '%s_fmin0.2_fmax10.0.csv'%(score_type)
elif ver == 3:
    fn_score_total = None
    
# Velocity Data & Parameters
# ---   ---   ---   ---   ---
#read profile information
fname_prof_info = '../../Data/site_reponse/vel_profs_ver%i/profile_info.csv'%(ver)

#filename for stationary and spatially varying model parameters
fname_param_model_stat = '../../Data/regression/model_stationary/all_trunc_stan_parameters.csv'
fname_param_model_svar = '../../Data/regression/model_spatially_varying/all_trunc_stan_parameters.csv'

#profile directories
dir_profs = '../../Data/site_reponse/vel_profs_ver%i/'%(ver)

# Output
# ---   ---   ---   ---   ---
#output directories
dir_out = '../../Data/site_reponse/postprocessing/'
dir_fig = dir_out + 'figures/'

#%% Load Data
### ======================================
#load profile information
df_prof_info = pd.read_csv(fname_prof_info)

#load scorring files
# binned
df_score_binned = {f_b: pd.read_csv(dir_score + fn_score_binned[f_b]) for f_b in fn_score_binned}
# total 
if not fn_score_total is None:
    df_score_total  = pd.read_csv(dir_score + fn_score_total)
else:
    df_score_total  = None

#read velocity profile locations
df_model_stat_param = pd.read_csv(fname_param_model_stat)
df_model_svar_param = pd.read_csv(fname_param_model_svar)

#%% Processing
### ======================================
#keep profile information for site response profiles
if not fn_score_total is None:
    df_score_total = pd.merge(df_score_total, df_prof_info[['DSID','VelID']], 
                              on=['DSID','VelID'], how='right')
#summarize binned information
df_score_binned = {f_b: pd.merge(df_score_binned[f_b], df_prof_info[['DSID','VelID']], 
                          on=['DSID','VelID'], how='right') 
                   for f_b in df_score_binned}

#keep stationary and spatially varying parms for site response profiles
df_model_stat_param = pd.merge(df_model_stat_param, df_prof_info[['DSID','VelID']], 
                               on=['DSID','VelID'], how='right')
df_model_svar_param = pd.merge(df_model_svar_param, df_prof_info[['DSID','VelID']], 
                               on=['DSID','VelID'], how='right')


#random realization column names
if ver == 1:
    #spatially varying - spatial realization
    i_svar_srlz = np.array([bool(re.match('^emp_svar_rlz(\d+)$', c)) for c in df_score_total.columns])
    cn_svar_srlz = df_score_total.columns[i_svar_srlz]
elif ver == 2:
    #stationery varying - depth realization
    i_stat_drlz  = np.array([bool(re.match('^emp_stat_drlz(\d+)$', c)) for c in df_score_total.columns])
    cn_stat_drlz = df_score_total.columns[i_stat_drlz]
    #spatially varying - spatial realization
    i_svar_srlz  = np.array([bool(re.match('^emp_svar_srlz(\d+)$', c)) for c in df_score_total.columns])
    cn_svar_srlz = df_score_total.columns[i_svar_srlz]
    #spatially varying - spatial & depth realization
    i_svar_srlz_drlz  = np.array([bool(re.match('^emp_svar_srlz(\d+)_drlz(\d+)$', c)) for c in df_score_total.columns])
    cn_svar_srlz_drlz = df_score_total.columns[i_svar_srlz_drlz]
elif ver == 3:
    #score dataframe columns
    score_columns = df_score_binned[list(fn_score_binned.keys())[0]].columns
    #stationery varying - depth realization
    i_stat_drlz  = np.array([bool(re.match('^emp_stat_drlz(\d+)$', c)) for c in score_columns])
    cn_stat_drlz = score_columns[i_stat_drlz]
    #spatially varying - spatial realization
    i_svar_srlz  = np.array([bool(re.match('^emp_svar_srlz(\d+)$', c)) for c in score_columns])
    cn_svar_srlz = score_columns[i_svar_srlz]
    #spatially varying - spatial & depth realization
    i_svar_srlz_drlz  = np.array([bool(re.match('^emp_svar_srlz(\d+)_drlz(\d+)$', c)) for c in score_columns])
    cn_svar_srlz_drlz = score_columns[i_svar_srlz_drlz]


#initialize vel profiles
df_vprof_emp_all  = list()
df_vprof_usgs_all = list()
df_vprof_stat_all, df_vprof_svar_all = list(), list()
#compute profile misfit
for k, vprof_info in df_prof_info.iterrows():
    #profile information
    fn_vprof_main = vprof_info.VelFName
    
    #read velocity profiles
    df_vprof_emp_all.append(  pd.read_csv(dir_profs + fn_vprof_main + '_emp'  + '.csv') )
    df_vprof_usgs_all.append( pd.read_csv(dir_profs + fn_vprof_main + '_usgs' + '.csv') )
    df_vprof_stat_all.append( pd.read_csv(dir_profs + fn_vprof_main + '_stat' + '.csv') )
    df_vprof_svar_all.append( pd.read_csv(dir_profs + fn_vprof_main + '_svar' + '.csv') )

    #compute rmse
    res_usgs = np.log( df_vprof_emp_all[k].Vs ) - np.log( df_vprof_usgs_all[k].Vs ) 
    res_stat = np.log( df_vprof_emp_all[k].Vs ) - np.log( df_vprof_stat_all[k].Vs ) 
    res_sver = np.log( df_vprof_emp_all[k].Vs ) - np.log( df_vprof_stat_all[k].Vs ) 
    #rmse summary
    df_prof_info.loc[k,'rmse_usgs'] = (res_usgs.values**2).mean()
    df_prof_info.loc[k,'rmse_stat'] = (res_stat.values**2).mean()
    df_prof_info.loc[k,'rmse_svar'] = (res_sver.values**2).mean()
    
#%% Output
### ======================================
#report score
for j, f_b in enumerate(fn_score_binned):
    #scorring arrays
    score_emp_usgs      = df_score_binned[f_b].emp_usgs.values
    score_emp_stat      = df_score_binned[f_b].emp_stat.values
    score_emp_svar      = df_score_binned[f_b].emp_svar.values
    score_emp_svar_srlz = df_score_binned[f_b].loc[:,cn_svar_srlz].values
    if ver > 1:
        score_emp_svar_srlz_drlz = df_score_binned[f_b].loc[:,cn_svar_srlz_drlz].values
        
    #print frequency score
    print("Frequency bin: %s"%f_b)
    print("\t USGS bias (mean, std): %.2f, %.2f"%(np.nanmean(score_emp_usgs), np.nanstd(score_emp_usgs)))
    print("\t Stat bias (mean, std): %.2f, %.2f"%(np.nanmean(score_emp_stat), np.nanstd(score_emp_stat)))
    print("\t Svar bias (mean, std): %.2f, %.2f"%(np.nanmean(score_emp_svar), np.nanstd(score_emp_svar)))
    print("\t Stat bias w/ var (mean, std): %.2f, %.2f"%(np.nanmean(score_emp_svar_srlz), np.nanstd(score_emp_svar_srlz)))
    if ver > 1:
        print("\t Svar bias w/ var (mean, std): %.2f, %.2f"%(np.nanmean(score_emp_svar_srlz_drlz), np.nanstd(score_emp_svar_srlz_drlz)))
            
#%% Plotting
### ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#color palete
cmap = plt.get_cmap("tab10")

# plot score vs vs30
# ---   ---   ---   ---
# plot binned scores
# - - - - - - - - - - - -
for j, f_b in enumerate(fn_score_binned):
    #vs30 array
    vs30_array = df_score_binned[f_b].Vs30.values
    #scorring arrays
    score_emp_usgs          = df_score_binned[f_b].emp_usgs.values
    score_emp_stat          = df_score_binned[f_b].emp_stat.values
    score_emp_svar          = df_score_binned[f_b].emp_svar.values
    score_emp_svar_srlz_mu  = df_score_binned[f_b].loc[:,cn_svar_srlz].mean(axis=1).values
    score_emp_svar_srlz_prc = df_score_binned[f_b].loc[:,cn_svar_srlz].quantile([.16, .84], axis=1).values
    if ver > 1:
        score_emp_stat_drlz_mu       = df_score_binned[f_b].loc[:,cn_stat_drlz].mean(axis=1).values
        score_emp_stat_drlz_prc      = df_score_binned[f_b].loc[:,cn_stat_drlz].quantile([.16, .84], axis=1).values
        score_emp_svar_srlz_drlz_mu  = df_score_binned[f_b].loc[:,cn_svar_srlz_drlz].mean(axis=1).values
        score_emp_svar_srlz_drlz_prc = df_score_binned[f_b].loc[:,cn_svar_srlz_drlz].quantile([.16, .84], axis=1).values
        
    #create figure score vs Vs30 - mean
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_vs30'%(score_type, f_b) 
    #usgs
    hl_usgs = ax.semilogx(vs30_array, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(vs30_array, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying (mean)
    hl_svar = ax.semilogx(vs30_array, score_emp_svar, 'd', color=cmap(0), label='Spatially Vayring Model')
    #edit properties
    ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
    ax.set_ylabel(r'Score',             fontsize=32)
    ax.legend(loc='lower right',        fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )

    #create figure score vs Vs30 - spatial variability
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_vs30_srlz'%(score_type, f_b) 
    #usgs
    hl_usgs = ax.semilogx(vs30_array, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(vs30_array, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying (mean)
    hl_svar_mu  = ax.semilogx(vs30_array, score_emp_svar_srlz_mu, 'd', color=cmap(0), label='Spatially Varying Model')
    #spatially varying (std)
    hl_svar_prc = ax.errorbar(vs30_array, y=score_emp_svar_srlz_mu, 
                              yerr=np.abs(score_emp_svar_srlz_prc - score_emp_svar_srlz_mu),
                              capsize=8, fmt='none', ecolor=hl_svar_mu[0].get_color(), linewidth=0.5)
    #edit properties
    ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
    ax.set_ylabel(r'Score',             fontsize=32)
    if j==0: ax.legend(loc='lower right',        fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )

    if ver > 1:
        #create figure score vs Vs30 - spatial & depth variability
        fig, ax = plt.subplots(figsize=(10,8))
        fname_fig = '%s_%s_vs30_srlz_drlz'%(score_type, f_b) 
        #usgs
        hl_usgs = ax.semilogx(vs30_array, score_emp_usgs, 'o', color='black', label='USGS')
        #stationary model (mean)
        hl_stat_mu = ax.semilogx(vs30_array, score_emp_stat_drlz_mu, 's', 
                                 color=cmap(1), label='Stationary Model')
        #spatially varying (std)
        hl_svar_prc = ax.errorbar(vs30_array, y=score_emp_stat_drlz_mu, 
                                  yerr=np.abs(score_emp_stat_drlz_prc - score_emp_stat_drlz_mu),
                                  capsize=8, fmt='none', ecolor=hl_stat_mu[0].get_color(), linewidth=0.5)
        #spatially varying (mean)
        hl_svar_mu  = ax.semilogx(vs30_array, score_emp_svar_srlz_drlz_mu, 'd', 
                                  color=cmap(0), label='Spatially Varying Model')
        #spatially varying (std)
        hl_svar_prc = ax.errorbar(vs30_array, y=score_emp_svar_srlz_drlz_mu, 
                                  yerr=np.abs(score_emp_svar_srlz_drlz_prc - score_emp_svar_srlz_drlz_mu),
                                  capsize=8, fmt='none', ecolor=hl_svar_mu[0].get_color(), linewidth=0.5)
        #edit properties
        ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
        ax.set_ylabel(r'Score',             fontsize=32)
        if j==0: ax.legend(loc='lower right',        fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.grid(which='both')
        ax.set_xlim([90, 2500])
        ax.set_ylim([-10, 10])
        fig.tight_layout()
        #save figure
        fig.savefig( dir_fig + fname_fig + '.png' )
        
        #create figure score vs Vs30 - variability effect
        fig, ax = plt.subplots(figsize=(10,8))
        fname_fig = '%s_%s_vs30_cmp'%(score_type, f_b) 
        #stationary model
        hl_stat    = ax.semilogx(vs30_array, score_emp_stat, 's', 
                                 color=cmap(1), label='Stat. Model - Mean')
        hl_stat_mu = ax.semilogx(vs30_array, score_emp_stat_drlz_mu, 'd', 
                                 color=cmap(1), label='Stat. Model - Varying')
        #spatially varying model
        hl_svar     = ax.semilogx(vs30_array, score_emp_svar, 's', 
                                  color=cmap(0), label='Spat. Vayring Model')
        hl_svar_mu  = ax.semilogx(vs30_array, score_emp_svar_srlz_drlz_mu, 'd', 
                                  color=cmap(0), label='Spat. Vayring Model - Varying')
        #edit properties
        ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
        ax.set_ylabel(r'Score',             fontsize=32)
        if j==0: ax.legend(loc='lower right',        fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.grid(which='both')
        ax.set_xlim([90, 2500])
        ax.set_ylim([-10, 10])
        fig.tight_layout()
        #save figure
        fig.savefig( dir_fig + fname_fig + '.png' )
    

# plot average scores
# - - - - - - - - - - - -
if not df_score_total is None:
    #vs30 array
    vs30_array = df_score_total.Vs30.values
    #scorring arrays
    score_emp_usgs          = df_score_total.emp_usgs.values
    score_emp_stat          = df_score_total.emp_stat.values
    score_emp_svar          = df_score_total.emp_svar.values
    score_emp_svar_srlz_mu  = df_score_total.loc[:,cn_svar_srlz].mean(axis=1).values
    score_emp_svar_srlz_prc = df_score_total.loc[:,cn_svar_srlz].quantile([.16, .84], axis=1).values
    if ver > 1:
        score_emp_stat_drlz_mu       = df_score_total.loc[:,cn_stat_drlz].mean(axis=1).values
        score_emp_stat_drlz_prc      = df_score_total.loc[:,cn_stat_drlz].quantile([.16, .84], axis=1).values
        score_emp_svar_srlz_drlz_mu  = df_score_total.loc[:,cn_svar_srlz_drlz].mean(axis=1).values
        score_emp_svar_srlz_drlz_prc = df_score_total.loc[:,cn_svar_srlz_drlz].quantile([.16, .84], axis=1).values
       
        #vs30 array
        vs30_array = df_score_binned[f_b].Vs30.values
        #scorring arrays
        score_emp_usgs          = df_score_binned[f_b].emp_usgs.values
        score_emp_stat          = df_score_binned[f_b].emp_stat.values
        score_emp_svar          = df_score_binned[f_b].emp_svar.values
        score_emp_svar_srlz_mu  = df_score_binned[f_b].loc[:,cn_svar_srlz].mean(axis=1).values
        score_emp_svar_srlz_prc = df_score_binned[f_b].loc[:,cn_svar_srlz].quantile([.16, .84], axis=1).values
        if ver > 1:
            score_emp_stat_drlz_mu       = df_score_binned[f_b].loc[:,cn_stat_drlz].mean(axis=1).values
            score_emp_stat_drlz_prc      = df_score_binned[f_b].loc[:,cn_stat_drlz].quantile([.16, .84], axis=1).values
            score_emp_svar_srlz_drlz_mu  = df_score_binned[f_b].loc[:,cn_svar_srlz_drlz].mean(axis=1).values
            score_emp_svar_srlz_drlz_prc = df_score_binned[f_b].loc[:,cn_svar_srlz_drlz].quantile([.16, .84], axis=1).values
            
    #create figure score vs Vs30 - mean
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_vs30'%(score_type, 'average') 
    #usgs
    hl_usgs = ax.semilogx(vs30_array, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(vs30_array, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying (mean)
    hl_svar = ax.semilogx(vs30_array, score_emp_svar, 'd', color=cmap(0), label='Spatially Varying Model')
    #edit properties
    ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
    ax.set_ylabel(r'Score',             fontsize=32)
    ax.legend(loc='lower right',        fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    #create figure score vs Vs30 - spatial variability
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_vs30_srlz'%(score_type, 'average') 
    #usgs
    hl_usgs = ax.semilogx(vs30_array, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(vs30_array, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying (mean)
    hl_svar_mu  = ax.semilogx(vs30_array, score_emp_svar_srlz_mu, 'd', color=cmap(0), label='Spatially Varying Model')
    #spatially varying (std)
    hl_svar_prc = ax.errorbar(vs30_array, y=score_emp_svar_srlz_mu, 
                              yerr=np.abs(score_emp_svar_srlz_prc - score_emp_svar_srlz_mu),
                              capsize=8, fmt='none', ecolor=hl_svar_mu[0].get_color(), linewidth=0.5)
    #edit properties
    ax.set_xlabel(r'$V_{S30}$ (m/sec)',  fontsize=32)
    ax.set_ylabel(r'Score',              fontsize=32)
    ax.legend(loc='lower right',         fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )
    
    if ver > 1:
        #create figure score vs Vs30 - spatial & depth variability
        fig, ax = plt.subplots(figsize=(10,8))
        fname_fig = '%s_%s_vs30_srlz_drlz'%(score_type, 'average') 
        #usgs
        hl_usgs = ax.semilogx(vs30_array, score_emp_usgs, 'o', color='black', label='USGS')
        #stationary model (mean)
        hl_stat_mu = ax.semilogx(vs30_array, score_emp_stat_drlz_mu, 's', 
                                 color=cmap(1), label='Stationary Model')
        #stationary varying (std)
        hl_svar_prc = ax.errorbar(vs30_array, y=score_emp_stat_drlz_mu, 
                                  yerr=np.abs(score_emp_stat_drlz_prc - score_emp_stat_drlz_mu),
                                  capsize=8, fmt='none', ecolor=hl_stat_mu[0].get_color(), linewidth=0.5)
        #spatially varying (mean)
        hl_svar_mu  = ax.semilogx(vs30_array, score_emp_svar_srlz_drlz_mu, 'd', 
                                  color=cmap(0), label='Spatially Varying Model')
        #spatially varying (std)
        hl_svar_prc = ax.errorbar(vs30_array, y=score_emp_svar_srlz_drlz_mu, 
                                  yerr=np.abs(score_emp_svar_srlz_drlz_prc - score_emp_svar_srlz_drlz_mu),
                                  capsize=8, fmt='none', ecolor=hl_svar_mu[0].get_color(), linewidth=0.5)
    
        #edit properties
        ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
        ax.set_ylabel(r'Score',             fontsize=32)
        ax.legend(loc='lower right',        fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.grid(which='both')
        ax.set_xlim([90, 2500])
        ax.set_ylim([-10, 10])
        fig.tight_layout()
        #save figure
        fig.savefig( dir_fig + fname_fig + '.png' )
    
        #create figure score vs Vs30 - variability effect
        fig, ax = plt.subplots(figsize=(10,8))
        fname_fig = '%s_%s_vs30_cmp'%(score_type, 'average') 
        #stationary model
        hl_stat    = ax.semilogx(vs30_array, score_emp_stat, 's', 
                                 color=cmap(1), label='Stationary Model - Mean')
        hl_stat_mu = ax.semilogx(vs30_array, score_emp_stat_drlz_mu, 'd', 
                                 color=cmap(1), label='Stationary Model - Varying')
        #spatially varying model
        hl_svar     = ax.semilogx(vs30_array, score_emp_svar, 's', 
                                  color=cmap(0), label='Spatially Varying Model - Mean')
        hl_svar_mu  = ax.semilogx(vs30_array, score_emp_svar_srlz_drlz_mu, 'd', 
                                  color=cmap(0), label='Spatially Varying Model - Varying')
        #edit properties
        ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
        ax.set_ylabel(r'Score',             fontsize=32)
        ax.legend(loc='lower right',        fontsize=30)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        ax.grid(which='both')
        ax.set_xlim([90, 2500])
        ax.set_ylim([-10, 10])
        fig.tight_layout()
        #save figure
        fig.savefig( dir_fig + fname_fig + '.png' )


# plot score vs rmse
# ---   ---   ---   ---
#rmse array
rmse_usgs = df_prof_info.rmse_usgs.values
rmse_stat = df_prof_info.rmse_stat.values
rmse_svar = df_prof_info.rmse_svar.values

# plot binned scores
# - - - - - - - - - - - -
for j, f_b in enumerate(fn_score_binned):
    #scorring arrays
    score_emp_usgs = df_score_binned[f_b].emp_usgs.values
    score_emp_stat = df_score_binned[f_b].emp_stat.values
    score_emp_svar = df_score_binned[f_b].emp_svar.values
    
    #create figure (score vs Vs30)
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_rmse'%(score_type, f_b) 
    #usgs
    hl_usgs = ax.semilogx(rmse_usgs, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(rmse_stat, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying (mean)
    hl_svar = ax.semilogx(rmse_svar, score_emp_svar, 'd', color=cmap(0), label='Spatially Varying Model')
    #edit properties
    ax.set_xlabel(r'RMSE',       fontsize=32)
    ax.set_ylabel(r'Score',      fontsize=32)
    if j==0: ax.legend(loc='lower right', fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    # ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )

# plot average scores
# - - - - - - - - - - - -
if not df_score_total is None:
    #scorring arrays
    score_emp_usgs = df_score_total.emp_usgs.values
    score_emp_stat = df_score_total.emp_stat.values
    score_emp_svar = df_score_total.emp_svar.values
    
    #create figure
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_rmse'%(score_type, 'average') 
    #usgs
    hl_usgs = ax.semilogx(rmse_usgs, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(rmse_stat, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying
    hl_svar = ax.semilogx(rmse_svar, score_emp_svar, 'd', color=cmap(0), label='Spatially Varying Model')
    #edit properties
    ax.set_xlabel(r'RMSE',       fontsize=32)
    ax.set_ylabel(r'Score',      fontsize=32)
    ax.legend(loc='lower right', fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    # ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )


# plot score vs fp
# ---   ---   ---   ---
#fp array
if ver <= 2:
    fp_array = 1/(4*df_prof_info.Dt.values)
else:
    fp_array = df_prof_info.fp.values

# plot binned scores
# - - - - - - - - - - - -
for j, f_b in enumerate(fn_score_binned):
    #scorring arrays
    score_emp_usgs = df_score_binned[f_b].emp_usgs.values
    score_emp_stat = df_score_binned[f_b].emp_stat.values
    score_emp_svar = df_score_binned[f_b].emp_svar.values
    
    #create figure (score vs Vs30)
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_fp'%(score_type, f_b) 
    #usgs
    hl_usgs = ax.semilogx(fp_array, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(fp_array, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying (mean)
    hl_svar = ax.semilogx(fp_array, score_emp_svar, 'd', color=cmap(0), label='Spatially Varying Model')
    #edit properties
    ax.set_xlabel(r'$f_p$ (hz)', fontsize=32)
    ax.set_ylabel(r'Score',      fontsize=32)
    if j==0: ax.legend(loc='lower right', fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    # ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )

# plot average scores
# - - - - - - - - - - - -
if not df_score_total is None:
    #scorring arrays
    score_emp_usgs = df_score_total.emp_usgs.values
    score_emp_stat = df_score_total.emp_stat.values
    score_emp_svar = df_score_total.emp_svar.values
    
    #create figure
    fig, ax = plt.subplots(figsize=(10,8))
    fname_fig = '%s_%s_fp'%(score_type, 'average') 
    #usgs
    hl_usgs = ax.semilogx(fp_array, score_emp_usgs, 'o', color='black', label='USGS')
    #stationary model
    hl_stat = ax.semilogx(fp_array, score_emp_stat, 's', color=cmap(1), label='Stationary Model')
    #spatially varying
    hl_svar = ax.semilogx(fp_array, score_emp_svar, 'd', color=cmap(0), label='Spatially Varying Model')
    #edit properties
    ax.set_xlabel(r'$f_p$ (hz)', fontsize=32)
    ax.set_ylabel(r'Score',      fontsize=32)
    ax.legend(loc='lower right', fontsize=30)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.grid(which='both')
    # ax.set_xlim([90, 2500])
    ax.set_ylim([-10, 10])
    #save figure
    fig.savefig( dir_fig + fname_fig + '.png' )
