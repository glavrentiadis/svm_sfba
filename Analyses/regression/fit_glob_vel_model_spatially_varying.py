#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 06:10:11 2022

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
from scipy import interpolate as interp
#statistics libraries
import pandas as pd
#plot libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import  AutoLocator as plt_autotick
import arviz as az
#stan library
import cmdstanpy
#user functions
sys.path.insert(0,'../python_lib/statistics')
sys.path.insert(0,'../python_lib/vel_prof')
sys.path.insert(0,'../python_lib/plotting')
from moving_mean import movingmean
from sigmoid import sigmoid
from calcvs0 import calcvs0
import pylib_contour_plots as pycplt

#plotting settings
#mpl.use('agg')

#%% Define Variables
### ======================================
#constants
z_star = 2.5

#original scaling coefficients from Shi and Asimaki (2018)
#Vs0 scaling
p1_orig = -2.1688*10**(-4)
p2_orig = 0.5182
p3_orig = 69.452
#k scaling
r1_orig =-59.67
r2_orig =-0.2722
r3_orig = 11.132
#n scaling
s1_orig = 4.110
s2_orig =-1.0521*10**(-4)
s3_orig =-10.827
s4_orig =-7.6187*10**(-3)

#proposed model (fixed coefficients) 
# # w/ outliers
# logVs30mid_fxd = 6.23704
# logVs30scl_fxd = 0.296187
# r1_fxd         =-2.34208
# r2_fxd         = 2.144715
# r3_fxd         = 0.3406835
# s2_fxd         = 5.321245
# #w/o outliers (partial)
# logVs30mid_fxd = 6.20947
# logVs30scl_fxd = 0.372021 
# r1_fxd         =-2.32772
# r2_fxd         = 3.73403
# r3_fxd         = 0.2734575
# s2_fxd         = 4.83542
#w/o outliers (partial, upd. priors)
# logVs30mid_fxd = 6.50456
# logVs30scl_fxd = 0.436805
# r1_fxd         =-2.296021
# r2_fxd         = 5.466964
# r3_fxd         = 0.423606
# s2_fxd         = 7.168569
# #w/o outliers
# logVs30mid_fxd = 6.14553
# logVs30scl_fxd = 0.384249
# r1_fxd         =-2.47132
# r2_fxd         = 2.88689
# r3_fxd         = 0.344808
# s2_fxd         = 3.928335
#w/o outliers (partial, v2)
logVs30mid_fxd = 6.49879
logVs30scl_fxd = 0.435501
r1_fxd         =-2.29844
r2_fxd         = 5.390775
r3_fxd         = 0.389704
s2_fxd         = 7.07134


#option for free of fixed scaling coefficients
flag_fxd_scl = True

#regression info
fname_stan_model = '../stan_lib/glob_reg_model_spatially_varying.stan'
#iteration samples
n_iter_warmup   = 50000
n_iter_sampling = 50000
# n_iter_warmup   = 100
# n_iter_sampling = 100
#MCMC parameters
n_chains        = 6
adapt_delta     = 0.8
max_treedepth   = 10


# Input/Output
# --------------------------------
#input flatfile
fname_flatfile = '../../Data/vel_profiles_dataset/all_velocity_profles.csv'

#flag truncate
flag_trunc_z1 = False
# flag_trunc_z1 = True

#Vs30 minimum threshold
flag_vs30_thres = True
Vs30_thres_min = 100

#outlier profile ids
flag_outlier_rm = True
# outlier_ds_id  = [3,  3,  3]
# outlier_vel_id = [56, 57, 58]
#outlier_ds_id  = [3,  3,  3,  3,  3,  3]
#outlier_vel_id = [56, 57, 43, 45, 58, 31]
outlier_ds_id  = [3,  3,  3,  3,  3,  3,  1]
outlier_vel_id = [56, 57, 58, 45, 43, 31, 9]

#output filename
# fname_out_main = 'all'
# fname_out_main = 'Jian'
fname_out_main = 'all_trunc'
#output directory
dir_out = '../../Data/regression/model_spatially_varying/'
dir_fig = dir_out + 'figures/'
#flag for paper quality paper
flag_paper = True

eps_lim =  [-2,2]

#%% Load Data
### ======================================
#load velocity profiles
df_velprofs = pd.read_csv(fname_flatfile)

#remove columns with nan mid-depth
df_velprofs = df_velprofs.loc[~np.isnan(df_velprofs.Depth_MPt),:]

#truncate profiles at depth of 1000m/sec
if flag_trunc_z1:
    df_velprofs = df_velprofs.loc[~df_velprofs.flag_Z1,:]

#truncate profiles at depth of 1000m/sec
if flag_vs30_thres:
    df_velprofs = df_velprofs.loc[df_velprofs.Vs30>=Vs30_thres_min,:]

#remove outliers
if flag_outlier_rm:
    #identify outliers
    # i_out = np.logical_and(np.isin(df_velprofs.DSID,  outlier_ds_id), 
    #                        np.isin(df_velprofs.VelID, outlier_vel_id))
    i_out = [np.logical_and(df_velprofs.DSID == ds_id, df_velprofs.VelID == vel_id).values for ds_id, vel_id in zip(outlier_ds_id, outlier_vel_id)]
    i_out = np.any(i_out, axis=0)
    print('removed %i datapoints as outliers'%i_out.sum())
    #remove outliers
    df_velprofs = df_velprofs.loc[~i_out,:]

#reset index
df_velprofs.reset_index(drop=True, inplace=True)

#%% Regression
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#unique datasets
ds_ids, ds_idx = np.unique(df_velprofs.DSID, return_index=True)
ds_names       = df_velprofs.DSName.iloc[ds_idx].values

#velocity profile ids
vel_id_dsid, vel_idx, vel_inv = np.unique(df_velprofs[['DSID','VelID']].values, axis=0, return_index=True, return_inverse=True)
vel_ids  = vel_inv + 1
n_vel    = vel_idx.shape[0]

# regression input data
# ---   ---   ---   ---
#prepare regression data
stan_data = {'N':       len(df_velprofs),
             'NVEL':    n_vel,
             'i_vel':   vel_ids,
             'z_star':  z_star,
             'Vs30':    df_velprofs.Vs30.values[vel_idx],
             'Z':       df_velprofs.Depth_MPt.values,
             'Y':       np.log(df_velprofs.Vs.values),
             'X':       df_velprofs[['X','Y']].values[vel_idx,:],
            }
if flag_fxd_scl:
    stan_data['logVs30mid_fxd'] = logVs30mid_fxd
    stan_data['logVs30scl_fxd'] = logVs30scl_fxd
    stan_data['s2_fxd']         = s2_fxd
    stan_data['r1_fxd']         = r1_fxd
    stan_data['r2_fxd']         = r2_fxd
    stan_data['r3_fxd']         = r3_fxd
#write as json file
fname_stan_data = dir_out +  'gobal_reg_stan_data' + '.json'
try:
    cmdstanpy.utils.jsondump(fname_stan_data, stan_data)
except AttributeError:
    cmdstanpy.utils.write_stan_json(fname_stan_data, stan_data)

# run stan
# ---   ---   ---   ---
#compile stan model
stan_model = cmdstanpy.CmdStanModel(stan_file=fname_stan_model) 
stan_model.compile(force=True)
#run full MCMC sampler
stan_fit = stan_model.sample(data=fname_stan_data, chains=n_chains, 
                             iter_warmup=n_iter_warmup, iter_sampling=n_iter_sampling,
                             seed=1, refresh=10, max_treedepth=max_treedepth, adapt_delta=adapt_delta,
                             show_progress=True,  output_dir=dir_out+'stan_fit/')

#delete json files
fname_dir = np.array( os.listdir(dir_out) )
#velocity filenames
fname_json = fname_dir[ [bool(re.search('\.json$',f_d)) for f_d in fname_dir] ]
for f_j in fname_json: os.remove(dir_out + f_j)


#%% Postprocessing
### ======================================
#initiaize flatfile for sumamry of non-erg coefficinets and residuals
df_velinfo  = df_velprofs[['DSID','DSName','VelID','VelName','Vs30','Z_max','Lat','Lon','X','Y','Depth_MPt','Thk', 'Vs', 'flag_Z1']]
df_profinfo = df_velprofs[['DSID','DSName','VelID','VelName','Vs30','Z_max','Lat','Lon','X','Y']].iloc[vel_idx,:].reset_index(drop=True)

# process regression output
# ---   ---   ---   ---
# Extract posterior samples
# - - - - - - - - - - - 
#hyper-parameters
col_names_hyp = ['logVs30mid','logVs30scl','r1','r2','r3','dr1','dr2','s2','sigma_vel','ell_rdB','omega_rdB']
#vel profile parameters
col_names_vs0 = ['Vs0.%i'%(k) for k in range(n_vel)]
col_names_k   = ['k.%i'%(k)   for k in range(n_vel)]
col_names_n   = ['n.%i'%(k)   for k in range(n_vel)]
col_names_rdB = ['rdB.%i'%(k)  for k in range(n_vel)]
col_names_all = col_names_hyp + col_names_vs0 + col_names_k + col_names_n + col_names_rdB
    
#extract raw hyper-parameter posterior samples
stan_posterior = np.stack([stan_fit.stan_variable(c_n) for c_n in col_names_hyp],  axis=1)
#vel profile parameters
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('Vs0_p')), axis=1)
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('k_p')),   axis=1)
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('n_p')),   axis=1)
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('rdB')),  axis=1)
    
#save raw-posterior distribution
df_stan_posterior_raw = pd.DataFrame(stan_posterior, columns = col_names_all)
df_stan_posterior_raw.to_csv(dir_out + fname_out_main + '_stan_posterior_raw' + '.csv', index=False)

# Summarize hyper-parameters
# - - - - - - - - - - - 
#summarize posterior distributions of hyper-parameters
perc_array        = np.array([0.05,0.25,0.5,0.75,0.95])
df_stan_hyp       = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_hyp.index = ['prc%.2f'%(prc) for prc in perc_array]
#add hyper-parameter mean
df_stan_hyp.loc['mean',:] = df_stan_posterior_raw[col_names_hyp].mean(axis = 0)
df_stan_hyp.to_csv(dir_out + fname_out_main + '_stan_hyperparameters' + '.csv', index=True)

#detailed posterior percentiles of posterior distributions
perc_array = np.arange(0.01,0.99,0.01)    
df_stan_posterior = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_posterior.index.name = 'prc'
df_stan_posterior.to_csv(dir_out + fname_out_main + '_stan_hyperposterior' + '.csv', index=True)

del stan_posterior, col_names_all

# Velocity profile parameters
# - - - - - - - - - - - 
#shape parameters
# Vs0
param_vs0_med = np.array([                 df_stan_posterior_raw.loc[:,f'Vs0.{k}'].median()   for k in range(n_vel)])
param_vs0_mu  = np.array([ np.exp( np.log( df_stan_posterior_raw.loc[:,f'Vs0.{k}'] ).mean() ) for k in range(n_vel)])
param_vs0_sig = np.array([ np.log(         df_stan_posterior_raw.loc[:,f'Vs0.{k}'] ).std()    for k in range(n_vel)])
param_vs0_p16 = np.array([ np.quantile(    df_stan_posterior_raw.loc[:,f'Vs0.{k}'], 0.16)     for k in range(n_vel)])
param_vs0_p84 = np.array([ np.quantile(    df_stan_posterior_raw.loc[:,f'Vs0.{k}'], 0.84)     for k in range(n_vel)])
# k
param_k_med = np.array([                 df_stan_posterior_raw.loc[:,f'k.{k}'].median()   for k in range(n_vel)])
param_k_mu  = np.array([ np.exp( np.log( df_stan_posterior_raw.loc[:,f'k.{k}'] ).mean() ) for k in range(n_vel)])
param_k_sig = np.array([         np.log( df_stan_posterior_raw.loc[:,f'k.{k}'] ).std()    for k in range(n_vel)])
param_k_p16 = np.array([ np.quantile(    df_stan_posterior_raw.loc[:,f'k.{k}'], 0.16)     for k in range(n_vel)])
param_k_p84 = np.array([ np.quantile(    df_stan_posterior_raw.loc[:,f'k.{k}'], 0.84)     for k in range(n_vel)])
# n
param_n_med = np.array([              df_stan_posterior_raw.loc[:,f'n.{k}'].median() for k in range(n_vel)])
param_n_mu  = np.array([              df_stan_posterior_raw.loc[:,f'n.{k}'].mean()   for k in range(n_vel)])
param_n_sig = np.array([              df_stan_posterior_raw.loc[:,f'n.{k}'].std()    for k in range(n_vel)])
param_n_p16 = np.array([ np.quantile( df_stan_posterior_raw.loc[:,f'n.{k}'], 0.16)   for k in range(n_vel)])
param_n_p84 = np.array([ np.quantile( df_stan_posterior_raw.loc[:,f'n.{k}'], 0.84)   for k in range(n_vel)])

#aleatory variability
param_sigma_vel_mu  = np.array([             df_stan_posterior_raw.sigma_vel.mean()]   * n_vel)
param_sigma_vel_med = np.array([             df_stan_posterior_raw.sigma_vel.median()] * n_vel)
param_sigma_vel_sig = np.array([             df_stan_posterior_raw.sigma_vel.std()]    * n_vel)
param_sigma_vel_p16 = np.array([np.quantile( df_stan_posterior_raw.sigma_vel, 0.16)]   * n_vel)
param_sigma_vel_p84 = np.array([np.quantile( df_stan_posterior_raw.sigma_vel, 0.84)]   * n_vel)

#between profile variability  (gaussian process)
# ell_rdB
param_ell_dBr_mu  = np.array([             df_stan_posterior_raw.ell_rdB.mean()]   * n_vel)
param_ell_dBr_med = np.array([             df_stan_posterior_raw.ell_rdB.median()] * n_vel)
param_ell_dBr_sig = np.array([             df_stan_posterior_raw.ell_rdB.std()]    * n_vel)
param_ell_dBr_p16 = np.array([np.quantile( df_stan_posterior_raw.ell_rdB, 0.16)]   * n_vel)
param_ell_dBr_p84 = np.array([np.quantile( df_stan_posterior_raw.ell_rdB, 0.84)]   * n_vel)
# omega_rdB
param_omega_dBr_mu  = np.array([             df_stan_posterior_raw.omega_rdB.mean()]   * n_vel)
param_omega_dBr_med = np.array([             df_stan_posterior_raw.omega_rdB.median()] * n_vel)
param_omega_dBr_sig = np.array([             df_stan_posterior_raw.omega_rdB.std()]    * n_vel)
param_omega_dBr_p16 = np.array([np.quantile( df_stan_posterior_raw.omega_rdB, 0.16)]   * n_vel)
param_omega_dBr_p84 = np.array([np.quantile( df_stan_posterior_raw.omega_rdB, 0.84)]   * n_vel)
#dB_r
param_dBr_med = np.array([              df_stan_posterior_raw.loc[:,f'rdB.{k}'].median() for k in range(n_vel)])
param_dBr_mu  = np.array([              df_stan_posterior_raw.loc[:,f'rdB.{k}'].mean()   for k in range(n_vel)])
param_dBr_sig = np.array([              df_stan_posterior_raw.loc[:,f'rdB.{k}'].std()    for k in range(n_vel)])
param_dBr_p16 = np.array([ np.quantile( df_stan_posterior_raw.loc[:,f'rdB.{k}'], 0.16)   for k in range(n_vel)])
param_dBr_p84 = np.array([ np.quantile( df_stan_posterior_raw.loc[:,f'rdB.{k}'], 0.84)   for k in range(n_vel)])

#summarize parameters
params_summary = np.vstack((param_vs0_mu,
                            param_k_mu, 
                            param_n_mu,
                            param_vs0_med,
                            param_k_med, 
                            param_n_med,
                            param_vs0_sig,
                            param_k_sig, 
                            param_n_sig,
                            param_vs0_p16,
                            param_k_p16, 
                            param_n_p16,
                            param_vs0_p84,
                            param_k_p84, 
                            param_n_p84,
                            param_dBr_mu,
                            param_dBr_med,
                            param_dBr_sig,
                            param_dBr_p16,
                            param_dBr_p84,
                            param_sigma_vel_mu,
                            param_sigma_vel_med,
                            param_sigma_vel_sig, 
                            param_sigma_vel_p16,
                            param_sigma_vel_p84,
                            param_ell_dBr_mu,
                            param_ell_dBr_med,
                            param_ell_dBr_sig, 
                            param_ell_dBr_p16,
                            param_ell_dBr_p84,
                            param_omega_dBr_mu,
                            param_omega_dBr_med,
                            param_omega_dBr_sig, 
                            param_omega_dBr_p16,
                            param_omega_dBr_p84 )).T
columns_names = ['param_vs0_mean',    'param_k_mean',    'param_n_mean',
                 'param_vs0_med',     'param_k_med',     'param_n_med',
                 'param_vs0_std',     'param_k_std',     'param_n_std',
                 'param_vs0_prc0.16', 'param_k_prc0.16', 'param_n_prc0.16',
                 'param_vs0_prc0.84', 'param_k_prc0.84', 'param_n_prc0.84',
                 'param_dBr_mean',    'param_dBr_med',   'param_dBr_std',   'param_dBr_prc0.16', 'param_dBr_prc0.84',              
                 'sigma_vel_mean',    'sigma_vel_med',   'sigma_vel_std',   'sigma_vel_prc0.16', 'sigma_vel_prc0.84',
                 'ell_dBr_mean',      'ell_dBr_med',     'ell_dBr_std',     'ell_dBr_prc0.16',   'ell_dBr_prc0.84',
                 'omega_dBr_mean',    'omega_dBr_med',   'omega_dBr_std',   'omega_dBr_prc0.16', 'omega_dBr_prc0.84']
df_params_summary = pd.DataFrame(params_summary, columns = columns_names, index=df_profinfo.index)

#create dataframe with parameters summary
df_params_summary = pd.merge(df_profinfo, df_params_summary, how='right', left_index=True, right_index=True)
df_params_summary[['DSID','VelID']] = df_params_summary[['DSID','VelID']].astype(int)
df_params_summary.to_csv(dir_out + fname_out_main + '_stan_parameters' + '.csv', index=False)

# Velocity profile prediction
# - - - - - - - - - - -
#mean scaling terms
logVs30mid_new = df_stan_hyp.loc['prc0.50','logVs30mid']
logVs30scl_new = df_stan_hyp.loc['prc0.50','logVs30scl']
#k scaling
r1_new = df_stan_hyp.loc['prc0.50','r1']
r2_new = df_stan_hyp.loc['prc0.50','r2']
r3_new = df_stan_hyp.loc['prc0.50','r3']
#n scaling
s2_new = df_stan_hyp.loc['prc0.50','s2']
#between event term
dB_r_new = param_dBr_med

#vs30 scaling
lnVs30s = (np.log(df_velprofs.Vs30.values)-logVs30mid_new)/logVs30scl_new

#mean profile parameters
param_n_new     =         1      + s2_new * sigmoid(lnVs30s)
param_a_new     =-1/param_n_new
param_k_new     = np.exp( r1_new + r2_new * sigmoid(lnVs30s) + r3_new * logVs30scl_new * np.log(1 + np.exp(lnVs30s)) )
param_vs0_new   = np.array([calcvs0(vs30, k, n, z_star) for (vs30, n, k) in zip(df_velprofs.Vs30.values, param_n_new, param_k_new)])
#considering between event effects
param_kdBr_new   = param_k_new * np.exp(dB_r_new[vel_inv] )
param_vs0dBr_new = np.array([calcvs0(vs30, k, n, z_star) for (vs30, n, k) in zip(df_velprofs.Vs30.values, param_n_new, param_kdBr_new)])

#orignal profile parameters
param_k_orig   = np.exp( r1_orig*(df_velprofs.Vs30.values)**r2_orig + r3_orig )
param_n_orig   = s1_orig*np.exp(s2_orig*df_velprofs.Vs30.values) + s3_orig*np.exp(s4_orig*df_velprofs.Vs30.values)
param_vs0_orig = p1_orig*(df_velprofs.Vs30.values)**2 + p2_orig*df_velprofs.Vs30.values + p3_orig

#mean prediction
y_data  = stan_data['Y']
y_new   = np.log(param_vs0_new    * ( 1 + param_k_new    * ( np.maximum(0, stan_data['Z']-z_star) ) )**(1/param_n_new))
y_newdB = np.log(param_vs0dBr_new * ( 1 + param_kdBr_new * ( np.maximum(0, stan_data['Z']-z_star) ) )**(1/param_n_new))
y_orig  = np.log(param_vs0_orig   * ( 1 + param_k_orig   * ( np.maximum(0, stan_data['Z']-z_star) ) )**(1/param_n_orig))
    
#compute residuals
res_tot     = y_data - y_new
res_orig    = y_data - y_orig
#within-event residuals
res_dW      = y_data - y_newdB
#between event residuals
res_dB      = res_tot - res_dW

#summary predictions and residuals
predict_summary = np.vstack((np.exp(y_newdB), np.exp(y_new), res_tot,res_dW, res_dB, res_orig)).T
columns_names   = ['VsProf_mean','VsProf_glob_mean','res_tot','res_dW','res_dB','res_orig']
df_predict_summary = pd.DataFrame(predict_summary, columns = columns_names, index=df_velprofs.index)
#create dataframe with predictions and residuals
df_predict_summary = pd.merge(df_velinfo, df_predict_summary, how='right', left_index=True, right_index=True)
df_predict_summary[['DSID','VelID']] = df_predict_summary[['DSID','VelID']].astype(int)
df_predict_summary.to_csv(dir_out + fname_out_main + '_stan_residuals' + '.csv', index=False)


#%% Comparison
### ======================================
# Total Residual
# ---------------------------
#residuals versus depth
i_sort    = np.argsort( df_predict_summary.Depth_MPt.values )
x_data    = df_predict_summary.Depth_MPt[i_sort]
y_data    = df_predict_summary.res_tot[i_sort]

#depth bins
x_bins = np.arange(0, 501, 100)
#binned residuals
x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

#residuals versus depth
fname_fig = (fname_out_main + '_total_residuals_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl1 = ax.plot(y_data,  x_data, 'o',  markersize=4,  color='gray',  fillstyle='none')
hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(y_mmed,  x_mbin, 's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('Total residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([-10, 500])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#total residuals versus depth (proposed and orignal model)
fname_fig = (fname_out_main + '_total_residuals_versus_depth_comparison').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_predict_summary.loc[:,'res_tot'], df_predict_summary.loc[:,'Depth_MPt'],  'o', markersize=4, label='Proposed Model')
hl = ax.plot(df_predict_summary.loc[:,'res_orig'], df_predict_summary.loc[:,'Depth_MPt'], 'o', label='Shi and Asimaki, 2018', zorder=1)
#edit properties
ax.set_xlabel('Total residuals',   fontsize=30)
ax.set_ylabel('Depth (m)',   fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim(eps_lim)
ax.set_ylim([0, 500])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Total residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#residuals versus Vs30
i_sort    = np.argsort( df_predict_summary.Vs30.values )
x_data    = df_predict_summary.Vs30[i_sort]
y_data    = df_predict_summary.res_tot[i_sort]

#vs30 bins
x_bins = np.logspace(np.log10(100),np.log10(2000),5)
#binned residuals
x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

#residuals versus Vs30
fname_fig = (fname_out_main + '_total_residuals_versus_Vs30').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl1 = ax.semilogy(y_data, x_data, 'o',  markersize=4,   color='gray',  fillstyle='none')
hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(y_mmed, x_mbin,  's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('Total residuals',    fontsize=30)
ax.set_ylabel(r'$V_{S30}~(m/sec)$', fontsize=30)
#ax.legend(loc='upper right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([100, 2500])
if not flag_paper:  ax.set_title(r'Total residuals versus $V_{S30}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#residuals versus Vs
i_sort    = np.argsort( df_predict_summary.Vs.values )
x_data    = df_predict_summary.Vs[i_sort]
y_data    = df_predict_summary.res_tot[i_sort]

#vs bins
x_bins = np.logspace(np.log10(100),np.log10(2000),8)
#binned residuals
x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

#residuals versus Vs
fname_fig = (fname_out_main + '_total_residuals_versus_Vs').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl1 = ax.semilogy(y_data, x_data, 'o',  markersize=4,   color='gray',  fillstyle='none')
hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(y_mmed, x_mbin,  's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('Total residuals',    fontsize=30)
ax.set_ylabel(r'$V_{S}~(m/sec)$', fontsize=30)
#ax.legend(loc='upper right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([100, 2500])
if not flag_paper:  ax.set_title(r'Total residuals versus $V_{S}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#residuals versus depth (Vs30 bins)
vs30_bins = [(0,100),(100,400),(400, 800),(800, 3000)]
for k, vs30_b in enumerate(vs30_bins):
    i_binned = np.logical_and(df_predict_summary.Vs30 >= vs30_b[0],
                                             df_predict_summary.Vs30 <  vs30_b[1])
    if i_binned.sum() == 0: continue
    df_predict_summ_binned = df_predict_summary.loc[i_binned,:].reset_index(drop=True)
    
    i_sort    = np.argsort( df_predict_summ_binned.Depth_MPt.values )
    x_data    = df_predict_summ_binned.Depth_MPt[i_sort]
    y_data    = df_predict_summ_binned.res_tot[i_sort]
    
    #depth bins
    x_bins = np.arange(0, 501, 100)
    #binned residuals
    x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

    fname_fig = (fname_out_main + '_total_residuals_versus_depth_vs30_%i_%i'%vs30_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.plot(y_data, x_data, 'o',  markersize=4,   color='gray',  fillstyle='none')
    hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
    hl3 = ax.plot(y_mmed, x_mbin,  's',  markersize=12, color='black', label='Median')
    hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                      capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                      label=r'$16-84^{th}$'+'\n Percentile')
    ax.set_xlabel('Total residuals',    fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.legend(loc='lower left', fontsize=30)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlim(eps_lim)
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    if not flag_paper: ax.set_title('Total residuals versus Depth \n $V_{S30}= [%.i, %.i)$ (m/sec)'%vs30_b, fontsize=30)
    fig.savefig( dir_fig + fname_fig + '.png' )

#total residuals versus depth (different Z_max bins)
zmax_bins = [(0,50),(50,100),(100, 250), (250, 500)]
for k, zmax_b in enumerate(zmax_bins):
    i_binned = np.logical_and(df_predict_summary.Z_max >= zmax_b[0],
                              df_predict_summary.Z_max <  zmax_b[1])
    if i_binned.sum() == 0: continue
    df_predict_summ_binned = df_predict_summary.loc[i_binned,:].reset_index(drop=True)
    
    i_sort    = np.argsort( df_predict_summ_binned.Depth_MPt.values )
    x_data    = df_predict_summ_binned.Depth_MPt[i_sort]
    y_data    = df_predict_summ_binned.res_tot[i_sort]
    
    #depth bins
    x_bins = np.arange(0, 501, 100)
    #binned residuals
    x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)
    
    fname_fig = (fname_out_main + '_total_residuals_versus_depth_zmax_%i_%i'%zmax_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.plot(y_data, x_data, 'o',  markersize=4,   color='gray',  fillstyle='none')
    hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
    hl3 = ax.plot(y_mmed, x_mbin,  's',  markersize=12, color='black', label='Median')
    hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                      capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                      label=r'$16-84^{th}$'+'\n Percentile')
    #edit properties
    ax.set_xlabel('Total residuals',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlim(eps_lim)
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    if not flag_paper: ax.set_title('Total Residuals versus Depth \n $z_{max}= [%.i, %.i)$ (m)'%zmax_b, fontsize=30)
    fig.savefig( dir_fig + fname_fig + '.png' )


# Within profile Residual
# ---------------------------
#within-profile residuals versus depth
i_sort    = np.argsort( df_predict_summary.Depth_MPt.values )
x_data    = df_predict_summary.Depth_MPt[i_sort]
y_data    = df_predict_summary.res_dW[i_sort]

#depth bins
x_bins = np.arange(0, 501, 100)
#binned residuals
x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl1 = ax.plot(y_data,  x_data, 'o',  markersize=4,  color='gray',  fillstyle='none')
hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(y_mmed,  x_mbin, 's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('Residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',  fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([-10, 500])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Within-profile Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#within-profile residuals versus depth (proposed and orignal model)
fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth_comparison').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_predict_summary.loc[:,'res_dW'], df_predict_summary.loc[:,'Depth_MPt'],  'o', markersize=4, label='Proposed Model')
hl = ax.plot(df_predict_summary.loc[:,'res_orig'], df_predict_summary.loc[:,'Depth_MPt'], 'o', label='Shi and Asimaki, 2018', zorder=1)
#edit properties
ax.set_xlabel('Residuals',   fontsize=30)
ax.set_ylabel('Depth (m)',   fontsize=30)
ax.legend(loc='lower right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([0, 500])
ax.invert_yaxis()
if not flag_paper: ax.set_title(r'Within-profile Residuals versus Depth', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#within-profile residuals versus Vs30
i_sort    = np.argsort( df_predict_summary.Vs30.values )
x_data    = df_predict_summary.Vs30[i_sort]
y_data    = df_predict_summary.res_dW[i_sort]

#vs30 bins
x_bins = np.logspace(np.log10(100),np.log10(2000),5)
#binned residuals
x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

#residuals versus Vs30
fname_fig = (fname_out_main + '_withinprof_residuals_versus_Vs30').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl1 = ax.semilogy(y_data, x_data, 'o',  markersize=4,   color='gray',  fillstyle='none')
hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(y_mmed, x_mbin,  's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('Residuals',  fontsize=30)
ax.set_ylabel(r'$V_{S30}~(m/sec)$', fontsize=30)
#ax.legend(loc='upper right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([100, 2500])
if not flag_paper: ax.set_title(r'Within-profile Residuals versus $V_{S30}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#within-profile residuals versus Vs
i_sort    = np.argsort( df_predict_summary.Vs.values )
x_data    = df_predict_summary.Vs[i_sort]
y_data    = df_predict_summary.res_dW[i_sort]

#vs30 bins
x_bins = np.logspace(np.log10(100),np.log10(2000),5)
#binned residuals
x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

#residuals versus Vs30
fname_fig = (fname_out_main + '_withinprof_residuals_versus_Vs').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl1 = ax.semilogy(y_data, x_data, 'o',  markersize=4,   color='gray',  fillstyle='none')
hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
hl3 = ax.plot(y_mmed, x_mbin,  's',  markersize=12, color='black', label='Median')
hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                  capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                  label=r'$16-84^{th}$'+'\n Percentile')
#edit properties
ax.set_xlabel('Residuals',  fontsize=30)
ax.set_ylabel(r'$V_{S} (m/sec)$', fontsize=30)
#ax.legend(loc='upper right', fontsize=30)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlim(eps_lim)
ax.set_ylim([100, 2500])
if not flag_paper: ax.set_title(r'Within-profile Residuals versus $V_{S}$', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#within-profile residuals versus depth (different Vs30 bins)
vs30_bins = [(0,100),(100,400),(400, 800),(800, 3000)]
for k, vs30_b in enumerate(vs30_bins):
    i_binned = np.logical_and(df_predict_summary.Vs30 >= vs30_b[0],
                                             df_predict_summary.Vs30 <  vs30_b[1])
    if i_binned.sum() == 0: continue
    df_predict_summ_binned = df_predict_summary.loc[i_binned,:].reset_index(drop=True)
    
    i_sort    = np.argsort( df_predict_summ_binned.Depth_MPt.values )
    x_data    = df_predict_summ_binned.Depth_MPt[i_sort]
    y_data    = df_predict_summ_binned.res_dW[i_sort]
    
    #depth bins
    x_bins = np.arange(0, 501, 100)
    #binned residuals
    x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

    fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth_vs30_%i_%i'%vs30_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.plot(y_data,  x_data, 'o',  markersize=4,  color='gray',  fillstyle='none')
    hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
    hl3 = ax.plot(y_mmed,  x_mbin, 's',  markersize=12, color='black', label='Median')
    hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                      capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                      label=r'$16-84^{th}$'+'\n Percentile')    
    #edit properties
    ax.set_xlabel('Residuals', fontsize=30)
    ax.set_ylabel('Depth (m)',              fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlim(eps_lim)
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    if not flag_paper: ax.set_title('Within-profile Residuals versus Depth \n $V_{S30}= [%.i, %.i)$ (m/sec)'%vs30_b, fontsize=30)
    fig.savefig( dir_fig + fname_fig + '.png' )

#total residuals versus depth (different Z_max bins)
zmax_bins = [(0,50),(50,100),(100, 250), (250, 500)]
for k, zmax_b in enumerate(zmax_bins):
    i_binned = np.logical_and(df_predict_summary.Z_max >= zmax_b[0],
                              df_predict_summary.Z_max <  zmax_b[1])
    if i_binned.sum() == 0: continue
    df_predict_summ_binned = df_predict_summary.loc[i_binned,:].reset_index(drop=True)
    
    i_sort    = np.argsort( df_predict_summ_binned.Depth_MPt.values )
    x_data    = df_predict_summ_binned.Depth_MPt[i_sort]
    y_data    = df_predict_summ_binned.res_dW[i_sort]
    
    #depth bins
    x_bins = np.arange(0, 501, 100)
    #binned residuals
    x_mbin, y_mmed, y_mmean, _, y_m16prc, y_m84prc = movingmean(y_data, x_data, x_bins)

    fname_fig = (fname_out_main + '_withinprof_residuals_versus_depth_zmax_%i_%i'%zmax_b).replace(' ','_')
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.plot(y_data,  x_data, 'o',  markersize=4,  color='gray',  fillstyle='none')
    hl2 = ax.plot(y_mmean, x_mbin, 'd',  markersize=12, color='black', label='Mean')
    hl3 = ax.plot(y_mmed,  x_mbin, 's',  markersize=12, color='black', label='Median')
    hl4 = ax.errorbar(y_mmed, x_mbin, xerr=np.abs(np.vstack((y_m16prc,y_m84prc)) - y_mmed),
                      capsize=8, fmt='none', ecolor=hl2[0].get_color(), linewidth=2,
                      label=r'$16-84^{th}$'+'\n Percentile')  
    #edit properties
    ax.set_xlabel('Residuals', fontsize=30)
    ax.set_ylabel('Depth (m)',              fontsize=30)
    ax.grid(which='both')
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlim(eps_lim)
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    if not flag_paper: ax.set_title('Within-profile Residuals versus Depth \n $z_{max}= [%.i, %.i)$ (m)'%zmax_b, fontsize=30)
    fig.savefig( dir_fig + fname_fig + '.png' )
    

# Parameter Scaling
# ---------------------------
vs30_array    = np.logspace(np.log10(100), np.log10(3000))
lnVs30s_array = (np.log(vs30_array)-logVs30mid_new) / logVs30scl_new
#scaling relationships
param_n_scl   =         1      + s2_new * sigmoid( lnVs30s_array )
param_k_scl   = np.exp( r1_new + r2_new * sigmoid( lnVs30s_array ) + r3_new * logVs30scl_new * np.log(1 + np.exp(lnVs30s_array)) )
param_a_scl   =-1/param_n_scl
param_vs0_scl = np.array([calcvs0(vs30, k, n, z_star) for (vs30, n, k) in zip(vs30_array, param_n_scl, param_k_scl)])


#scaling k
fname_fig = (fname_out_main + '_scaling_param_k').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.loglog(vs30_array, param_k_scl, '-', linewidth=4, zorder=10, color='k', label='Spatially\nVarying Model')
hl = ax.loglog(df_params_summary.Vs30, df_params_summary.param_k_med, 'o',  linewidth=2, color='gray', label='Posterior Median')
hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_k_med,
                  yerr=np.abs(df_params_summary[['param_k_prc0.16','param_k_prc0.84']].values - df_params_summary.param_k_med.values[:,np.newaxis]).T, 
                  capsize=4, fmt='none', ecolor='gray', label='Posterior Std')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
ax.set_ylabel(r'$k$',               fontsize=32)
ax.legend(loc='upper left', fontsize=32)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_ylim([5e-2, 50])
if not flag_paper: ax.set_title(r'$k$ Scaling', fontsize=32)
fig.tight_layout()
fig.savefig(dir_fig+fname_fig+'.png', bbox_inches='tight')

#scaling n
fname_fig = (fname_out_main + '_scaling_param_n').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.semilogx(vs30_array, param_n_scl, '-', linewidth=4, zorder=10, color='k', label='Spatially\nVarying Model')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
ax.set_ylabel(r'$n$',               fontsize=32)
# ax.legend(loc='upper left', fontsize=32)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_yticks(np.arange(11))
ax.set_ylim([0, 10.5])
if not flag_paper: ax.set_title(r'$n$ Scaling', fontsize=32)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


#scaling Vs0
fname_fig = (fname_out_main + '_scaling_param_vs0').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl_scl, = ax.loglog(vs30_array, param_vs0_scl, '-', linewidth=4, zorder=10, color='k', label='Spatially\nVarying Model')
hl_med, = ax.loglog(df_params_summary.Vs30, df_params_summary.param_vs0_med, 'o',  linewidth=2, color='gray', label='Posterior Median')
hl_std = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_vs0_med, 
                      yerr=np.abs(df_params_summary[['param_vs0_prc0.16','param_vs0_prc0.84']].values - df_params_summary.param_vs0_med.values[:,np.newaxis]).T, 
                      capsize=4, fmt='none', ecolor='gray', label='Posterior Std')
hl_ln,  = ax.loglog([100,2000],[100,2000], linestyle=':', linewidth=3, color='black', label='1:1 Line')
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=30)
ax.set_ylabel(r'$V_{S0}$ (m/sec)',  fontsize=30)
# ax.legend(handles=[hl_scl,hl_med,hl_std,hl_ln], loc='lower right', fontsize=32)
ax.legend(handles=[hl_scl,hl_med,hl_std,hl_ln], loc='upper left', fontsize=32)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
if not flag_paper: ax.set_title(r'$V_{S0}$ Scaling', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#delta B r
fname_fig = (fname_out_main + '_scatter_param_dBr').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
hl = ax.plot(df_params_summary.Vs30, df_params_summary.param_dBr_med, 'o',  linewidth=2, color='gray', label='Posterior Median')
hl = ax.errorbar(df_params_summary.Vs30, df_params_summary.param_dBr_med, 
                 yerr=np.abs(df_params_summary[['param_dBr_prc0.16','param_dBr_prc0.84']].values - df_params_summary.param_dBr_med.values[:,np.newaxis]).T, 
                 capsize=4, fmt='none', ecolor='gray', label='Posterior Std')                  
#edit properties
ax.set_xlabel(r'$V_{S30}$ (m/sec)', fontsize=32)
ax.set_ylabel(r'$\delta B_r$',      fontsize=32)
ax.legend(loc='lower right', fontsize=32)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
ax.set_ylim([-10, 10])
if not flag_paper: ax.set_title(r'$\delta B_r$ Scatter ', fontsize=30)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )


# Spatial variability
# ---------------------------
# pl_win = [[36, 39], [-124, -121]]
pl_win = [[36.5, 38.75], [-123.25, -121.25]]

#median delta B r
data2plot = df_params_summary[['Lat','Lon','param_dBr_med']].values

fname_fig =  (fname_out_main + '_locations_param_dBr_med').replace(' ','_')
fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=-0.5,  cmax=0.5, flag_grid=False, 
                                                      title=None, cbar_label='', log_cbar = False, 
                                                      frmt_clb = '%.2f', alpha_v = 0.7, cmap='seismic', marker_size=70.)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'Median: $\delta B_r$', size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
if not flag_paper: ax.set_title(r'$\delta B_r$ Spatial Variability ', fontsize=30)
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
data2plot = df_params_summary[['Lat','Lon','param_dBr_std']].values

fname_fig =  (fname_out_main + '_locations_param_dBr_std').replace(' ','_')
fig, ax, cbar, data_crs, gl = pycplt.PlotScatterCAMap(data2plot, cmin=0,  cmax=.5, flag_grid=False, 
                                                      title=None, cbar_label='', log_cbar = False, 
                                                      frmt_clb = '%.2f', alpha_v = 0.7, cmap='Reds', marker_size=70.)
#edit figure properties
cbar.ax.tick_params(labelsize=28)
cbar.set_label(r'Standard Deviation: $\delta B_r$', size=30)
ax.set_xlim(pl_win[1])
ax.set_ylim(pl_win[0])
if not flag_paper: ax.set_title(r'$\delta B_r$ Spatial Variability ', fontsize=30)
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


# Iterate Profiles
# ---------------------------
#iterate over vel profiles
for k, v_id_dsid in enumerate(vel_id_dsid):
    print('Plotting vel. profile %i of %i'%(k+1, n_vel))
    #extract vel profile of interest
    df_vel = df_predict_summary.loc[np.logical_and(df_velprofs.DSID==v_id_dsid[0], df_velprofs.VelID==v_id_dsid[1]), ]
    #profile name
    name_vel =  df_vel.DSName.values[0]+' '+df_vel.VelName.values[0]
    fname_vel = (fname_out_main + '_vel_prof_' + name_vel).replace(' ','_')

    #velocity prof information
    vs30    = df_vel.Vs30.values[0]
    lnVs30s = (np.log(vs30)-logVs30mid_new)/logVs30scl_new
    param_n      =         1      + s2_new * sigmoid(lnVs30s)
    param_k      = np.exp( r1_new + r2_new * sigmoid(lnVs30s) + r3_new * logVs30scl_new * np.log( 1 + np.exp(lnVs30s) ) )
    param_a      =-1/param_n
    param_vs0    = calcvs0(vs30, param_k, param_n, z_star)
    param_kdBr   = param_k * np.exp( dB_r_new[k] )
    param_vs0dBr = calcvs0(vs30, param_kdBr, param_n, z_star)

    #velocity profile
    depth_array = np.concatenate([[0], np.cumsum(df_vel.Thk)])
    vel_array   = np.concatenate([df_vel.Vs.values, [df_vel.Vs.values[-1]]])
    #velocity model
    depth_m_array = np.linspace(depth_array.min(), depth_array.max(), 1000)
    vel_m_array   = param_vs0    * ( 1 + param_k    * ( np.maximum(0, depth_m_array-z_star) ) )**(1/param_n)
    vel_mdB_array = param_vs0dBr * ( 1 + param_kdBr * ( np.maximum(0, depth_m_array-z_star) ) )**(1/param_n)
    
    #create figure   
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.step(vel_array,     depth_array,   color='gray', label='Velocity Profile')
    hl2 = ax.plot(vel_m_array,   depth_m_array, linewidth=2.0, linestyle='dashed', color='k', label=r'Velocity Model (w/o $\delta B_r$)')
    hl3 = ax.plot(vel_mdB_array, depth_m_array, linewidth=2.0,                     color='k', label=r'Velocity Model (w/ $\delta B_r$)')
    #edit properties
    ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.legend(loc='lower left', fontsize=30)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 500])
    ax.invert_yaxis()
    ax.set_title(name_vel, fontsize=30)
    fig.tight_layout()
    fig.savefig( dir_fig + fname_vel + '.png' )
    

# Summary regression
# ---------------------------
#save summary statistics
stan_summary_fname = dir_out  + fname_out_main + '_stan_summary' + '.txt'
with open(stan_summary_fname, 'w') as f:
    print(stan_fit, file=f)

#create and save trace plots
dir_fig = dir_out  + 'summary_figs/'
#create figures directory if doesn't exit
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#create stan trace plots
stan_az_fit = az.from_cmdstanpy(stan_fit)
#stan_az_fit = az.InferenceData(posterior=stan_fit.draws_xr())
for c_name in col_names_hyp:
    #create trace plot with arviz
    ax = az.plot_trace(stan_az_fit,  var_names=c_name, figsize=(10,5) ).ravel()
    ax[0].yaxis.set_major_locator(plt_autotick())
    ax[0].set_xlabel('sample value')
    ax[0].set_ylabel('frequency')
    ax[0].set_title('')
    ax[0].grid(axis='both')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('sample value')
    ax[1].grid(axis='both')
    ax[1].set_title('')
    fig = ax[0].figure
    fig.suptitle(c_name)
    fig.savefig(dir_fig + fname_out_main + '_stan_traceplot_' + c_name + '_arviz' + '.png')

#summary residuals
print('Total Residuals Mean: %.2f'%df_predict_summary.loc[:,'res_tot'].mean())
print('Total Residuals Std Dev: %.2f'%df_predict_summary.loc[:,'res_tot'].std())
print('Within-profile Residuals Mean: %.2f'%df_predict_summary.loc[:,'res_dW'].mean())
print('Within-profile Residuals Std Dev: %.2f'%df_predict_summary.loc[:,'res_dW'].std())

f = open(dir_out + fname_out_main + '_stan_summary_regression' + '.txt', 'w')
f.write('Total Residuals Mean: %.2f\n'%df_predict_summary.loc[:,'res_tot'].mean())
f.write('Total Residuals Std Dev: %.2f\n'%df_predict_summary.loc[:,'res_tot'].std())
f.write('Within-profile Residuals Mean: %.2f\n'%df_predict_summary.loc[:,'res_dW'].mean())
f.write('Within-profile Residuals Std Dev: %.2f\n'%df_predict_summary.loc[:,'res_dW'].std())
f.close()
