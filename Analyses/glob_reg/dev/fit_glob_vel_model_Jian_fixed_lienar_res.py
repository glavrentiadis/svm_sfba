#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 06:10:11 2022

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
import arviz as az
mpl.use('agg')
#stan library
import cmdstanpy

#%% Define Variables
### ======================================

z_star = 2.5

#regression info
fname_stan_model = '../stan_lib/jian_fun_glob_fixed_reg_linear_res.stan'
#iteration samples
n_iter_warmup   = 500
n_iter_sampling = 500
#MCMC parameters
n_chains        = 4
adapt_delta     = 0.8
max_treedepth   = 10


# Input/Output
# --------------------------------
#input flatfile
fname_flatfile = '../../Data/vel_profiles/all_velocity_profles.csv'
# fname_flatfile = '../../Data/vel_profiles/Jian_velocity_profles.csv'

#flag truncate
# flag_trunc_z1 = False
flag_trunc_z1 = True

#output filename
# fname_out_main = 'all'
# fname_out_main = 'Jian'
fname_out_main = 'all_trunc'
#output directory
dir_out = '../../Data/global_reg/bayesian_fit/JianFunFixed_linear_res/' + fname_out_main + '/'
dir_fig = dir_out + 'figures/'

#%% Load Data
### ======================================
#load velocity profiles
df_velprofs = pd.read_csv(fname_flatfile)

if flag_trunc_z1:
    df_velprofs = df_velprofs.loc[~df_velprofs.flag_Z1,:]

#%% Run Regression
### ======================================
#create output directory
pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 
pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True) 

#unique datasets
ds_ids, ds_idx = np.unique(df_velprofs.DSID, return_index=True)
ds_names       = df_velprofs.DSName.iloc[ds_idx].values

#velocity profile ids
vel_ids, vel_idx, vel_inv = np.unique(df_velprofs[['DSID','VelID']].values, axis=0, return_index=True, return_inverse=True)
# vel_ids = vel_ids[:100, ]
vel_ids  = vel_inv + 1
n_vel    = vel_ids.shape[0]

# regression input data
# ---   ---   ---   ---
#prepare regression data
stan_data = {'N':       len(df_velprofs),
             'NVEL':    n_vel,
             'i_vel':   vel_ids,
             'z_star':  z_star,
             'Vs30':    df_velprofs.Vs30.values,
             'Z':       df_velprofs.Depth_MPt.values,
             'Y':       df_velprofs.Vs.values,
            }
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
                             seed=1, max_treedepth=max_treedepth, adapt_delta=adapt_delta,
                             show_progress=True,  output_dir=dir_out+'stan_fit/')

#delete json files
fname_dir = np.array( os.listdir(dir_out) )
#velocity filenames
fname_json = fname_dir[ [bool(re.search('\.json$',f_d)) for f_d in fname_dir] ]
for f_j in fname_json: os.remove(dir_out + f_j)

#%% Postprocessing
### ======================================
# process regression output
# ---   ---   ---   ---
# Extract posterior samples
# - - - - - - - - - - - 
#hyper-parameters
col_names_hyp = ['sigma_Vs0','sigma_vel']
#between terms
col_names_dB     = ['dB_Vs0.%i'%(k)       for k in range(n_vel)]
col_names_all = col_names_hyp + col_names_dB
    
#extract raw hyper-parameter posterior samples
stan_posterior = np.stack([stan_fit.stan_variable(c_n) for c_n in col_names_hyp], axis=1)
#extract raw between event posterior samples
stan_posterior = np.concatenate((stan_posterior, stan_fit.stan_variable('dB_Vs0')),     axis=1)

#save raw-posterior distribution
df_stan_posterior_raw = pd.DataFrame(stan_posterior, columns = col_names_all)
df_stan_posterior_raw.to_csv(dir_out + fname_out_main + '_stan_posterior_raw' + '.csv', index=False)

# Summarize hyper-parameters
# - - - - - - - - - - - 
#summarize posterior distributions of hyper-parameters
perc_array = np.array([0.05,0.25,0.5,0.75,0.95])
df_stan_hyp = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_hyp = df_stan_hyp.append(df_stan_posterior_raw[col_names_hyp].mean(axis = 0), ignore_index=True)
df_stan_hyp.index = ['prc_%.2f'%(prc) for prc in perc_array]+['mean'] 
df_stan_hyp.to_csv(dir_out + fname_out_main + '_stan_hyperparameters' + '.csv', index=True)

#detailed posterior percentiles of posterior distributions
perc_array = np.arange(0.01,0.99,0.01)    
df_stan_posterior = df_stan_posterior_raw[col_names_hyp].quantile(perc_array)
df_stan_posterior.index.name = 'prc'
df_stan_posterior.to_csv(dir_out + fname_out_main + '_stan_hyperposterior' + '.csv', index=True)

del col_names_hyp, col_names_dB
del stan_posterior, col_names_all

# Velocity profile parameters
# - - - - - - - - - - - 
#initiaize flatfile for sumamry of non-erg coefficinets and residuals
df_velinfo =  df_velprofs[['DSID','DSName','VelID','VelName','Vs30','Lat','Lon']]
    
# Vs0
param_vs0_mu  = np.array([df_stan_posterior_raw.loc[:,f'Vs0.{k}'].mean()   for k in range(n_vel)])
param_vs0_mu  = param_vs0_mu[vel_inv]
param_vs0_med = np.array([df_stan_posterior_raw.loc[:,f'Vs0.{k}'].median() for k in range(n_vel)])
param_vs0_med = param_vs0_med[vel_inv]
param_vs0_sig = np.array([df_stan_posterior_raw.loc[:,f'Vs0.{k}'].std()    for k in range(n_vel)])
param_vs0_sig = param_vs0_sig[vel_inv]
# k
param_k_mu  = np.array([df_stan_posterior_raw.loc[:,f'k.{k}'].mean()   for k in range(n_vel)])
param_k_mu  = param_k_mu[vel_inv]
param_k_med = np.array([df_stan_posterior_raw.loc[:,f'k.{k}'].median() for k in range(n_vel)])
param_k_med = param_k_med[vel_inv]
param_k_sig = np.array([df_stan_posterior_raw.loc[:,f'k.{k}'].std()    for k in range(n_vel)])
param_k_sig = param_k_sig[vel_inv]
# n
param_n_mu  = np.array([df_stan_posterior_raw.loc[:,f'n.{k}'].mean()   for k in range(n_vel)])
param_n_mu  = param_n_mu[vel_inv]
param_n_med = np.array([df_stan_posterior_raw.loc[:,f'n.{k}'].median() for k in range(n_vel)])
param_n_med = param_n_med[vel_inv]
param_n_sig = np.array([df_stan_posterior_raw.loc[:,f'n.{k}'].std()    for k in range(n_vel)])
param_n_sig = param_n_sig[vel_inv]

# Velocity profile prediction
# - - - - - - - - - - - 
#mean prediction
y_data = stan_data['Y']
y_mu   = param_vs0_mu * ( 1 + param_k_mu*( np.maximum(0, stan_data['Z']-z_star) ) )**param_n_mu
    
#compute residuals
res_tot     = y_data - y_mu
res_between = [df_stan_posterior_raw.loc[:,f'dB_Vs0.{k}'].mean() for k in range(n_vel)]
res_between = np.array([res_between[k] for k in (vel_inv).astype(int)])
res_within  = res_tot - res_between

#summary predictions and residuals
predict_summary = np.vstack((y_mu, res_tot, res_between, res_within)).T
columns_names   = ['vel_prof_mu','res_tot','res_between','res_within']
df_predict_summary = pd.DataFrame(predict_summary, columns = columns_names, index=df_velprofs.index)
#create dataframe with predictions and residuals
df_predict_summary = pd.merge(df_velinfo, df_predict_summary, how='right', left_index=True, right_index=True)
df_predict_summary[['DSID','VelID']] = df_predict_summary[['DSID','VelID']].astype(int)
df_predict_summary.to_csv(dir_out + fname_out_main+ '_stan_residuals' + '.csv', index=True)






#initalize paramters
df_param_all = df_velprofs[['DSID','DSName','VelID','VelName','Vs30','Lat','Lon']].iloc[vel_idx,:].reset_index(drop=True)

# Iterate over profiles
# --------------------------------
for k, v_id in enumerate(vel_ids):
    print('Regressing profile %i of %i'%(k+1, n_vel))
    #extract vel profile of interest
    df_vel = df_velprofs.loc[np.logical_and(df_velprofs.DSID==v_id[0], df_velprofs.VelID==v_id[1]), ]
    #fitted profile
    name_vel =  df_vel.DSName.values[0]+' '+df_vel.VelName.values[0]
    fname_vel = (fname_out_main + '_vel_prof_' + name_vel).replace(' ','_')
   
    Vs30_vel = df_vel.Vs30.values[0]
    print(Vs30_vel )
   
    # compute fitted model
    # ---   ---   ---   ---    
    #mean parameters
    param_vs0 = p1_mu*(Vs30_vel)**2 + p2_mu*Vs30_vel + p3_mu
    param_k   = np.exp( r1_mu*(Vs30_vel)**r2_mu + r3_mu )
    param_n   = 1/( s1_mu*np.exp(s2_mu*Vs30_vel) + s3_mu*np.exp(s4_mu*Vs30_vel) ) 
    
    #mean model and residuals
    vs_model = param_vs0 * np.maximum(1, (1 + param_k*(df_vel.Depth.values-z_star))**param_n )
    df_vel.loc[df_vel.index,'Vs_model'] = vs_model 
    df_vel.loc[df_vel.index,'res']      = df_vel.Vs.values - vs_model 
    df_velprofs.loc[df_vel.index,'Vs_model'] = vs_model 
    df_velprofs.loc[df_vel.index,'res']      = df_vel.Vs.values - vs_model 

    #profile misfit statistics
    vel_misfit_mean = np.mean(df_vel.loc[df_vel.index,'res'] )
    vel_misfit_std  = np.std(df_vel.loc[df_vel.index,'res'] )

    #summarize vel parameters
    df_param_all.loc[k,['Vs0','k','n','mean_res','std_res']] = [param_vs0,param_k,param_n,vel_misfit_mean,vel_misfit_std]

    # figures
    # ---   ---   ---   ---
    #create figure   
    fig, ax = plt.subplots(figsize = (10,10))
    hl1 = ax.plot(df_vel.Vs,       df_vel.Depth, linewidth=2.0, label='Original profile')
    hl2 = ax.plot(df_vel.Vs_model, df_vel.Depth, linewidth=2.0, label='Velocity model')
    #edit properties
    ax.invert_yaxis()
    ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    ax.legend(loc='upper right', fontsize=30)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.set_title(name_vel, fontsize=22)
    fig.tight_layout()
    fig.savefig( dir_fig + fname_vel + '.png' )



# Summary
# --------------------------------
# output
# ---   ---   ---   ---
#regression model and residuals
df_velprofs.to_csv(dir_out + fname_out_main + '_stan_residuals' + '.csv', index=True)
#regression parameters
df_param_all.to_csv(dir_out + fname_out_main + '_stan_parameters' + '.csv', index=True)

print('Residuals Mean: %.1f'%df_velprofs.res.mean())
print('Residuals Std Dev: %.1f'%df_velprofs.res.std())


# figures
# ---   ---   ---   ---
# cmpare model parameters versus vs30
#Vs0
fname_fig = (fname_out_main + 'param_Vs0').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
#ploting
for d_id, d_name in zip(ds_ids, ds_names):
    i_ds = df_param_all.DSID.values == d_id
    hl = ax.plot(df_param_all.loc[i_ds,'Vs30'], df_param_all.loc[i_ds,'Vs0'], 'o', label=d_name)
#edit properties
ax.set_xlabel('$V_{S30}$ (m/sec)',  fontsize=30)
ax.set_ylabel('$V_{S0}$ (m/sec)',  fontsize=30)
ax.grid(which='both')
ax.legend(loc='upper right', fontsize=30)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#k
fname_fig = (fname_out_main + 'param_k').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
#ploting
for d_id, d_name in zip(ds_ids, ds_names):
    i_ds = df_param_all.DSID.values == d_id
    hl = ax.plot(df_param_all.loc[i_ds,'Vs30'], df_param_all.loc[i_ds,'k'], 'o', label=d_name)
#edit properties
ax.set_xlabel('$V_{S30}$ (m/sec)',  fontsize=30)
ax.set_ylabel('$k$',  fontsize=30)
ax.grid(which='both')
ax.legend(loc='upper right', fontsize=30)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#n
fname_fig = (fname_out_main + 'param_n').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
#ploting
for d_id, d_name in zip(ds_ids, ds_names):
    i_ds = df_param_all.DSID.values == d_id
    hl = ax.plot(df_param_all.loc[i_ds,'Vs30'], df_param_all.loc[i_ds,'n'], 'o', label=d_name)
#edit properties
ax.set_xlabel('$V_{S30}$ (m/sec)',  fontsize=30)
ax.set_ylabel('$n$',  fontsize=30)
ax.grid(which='both')
ax.legend(loc='upper right', fontsize=30)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#sigma
fname_fig = (fname_out_main + 'param_sig').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
#ploting
for d_id, d_name in zip(ds_ids, ds_names):
    i_ds = df_param_all.DSID.values == d_id
    hl = ax.plot(df_param_all.loc[i_ds,'Vs30'], df_param_all.loc[i_ds,'sigma'], 'o', label=d_name)
#edit properties
ax.set_xlabel('$V_{S30}$ (m/sec)',  fontsize=30)
ax.set_ylabel('$\sigma$ (m/sec)',  fontsize=30)
ax.grid(which='both')
ax.legend(loc='upper right', fontsize=30)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )

#residuals
fname_fig = (fname_out_main + '_residuals_versus_depth').replace(' ','_')
fig, ax = plt.subplots(figsize = (10,10))
for d_id, d_name in zip(ds_ids, ds_names):
    i_ds = df_velprofs.DSID.values == d_id
    hl = ax.plot(df_velprofs.loc[i_ds,'res'], df_velprofs.loc[i_ds,'Depth'], 'o', label=d_name)
#edit properties
ax.invert_yaxis()
ax.set_xlabel('residuals',  fontsize=30)
ax.set_ylabel('Depth (m)',      fontsize=30)
ax.grid(which='both')
ax.legend(loc='lower right', fontsize=30)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.set_xlim([-1000, 1000])
fig.tight_layout()
fig.savefig( dir_fig + fname_fig + '.png' )