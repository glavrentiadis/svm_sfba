#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 18:06:14 2025

@author: glavrent
"""


#load libraries
import os
import pathlib
import sys
#arithmetic libraries
import numpy  as np
from scipy import interpolate as interp
#dataframe libraries
import pandas as pd
import geopandas as gpd
#ploting libraries
import matplotlib.pyplot as plt
#user functions
sys.path.insert(0,'../python_lib/statistics')
sys.path.insert(0,'../python_lib/vel_prof')
sys.path.insert(0,'../python_lib/plotting')
sys.path.insert(0,'../implementation')
sys.path.insert(0,'../miscellaneous')
sys.path.insert(0,'../python_lib/usgs')
#vs30 map
from query_willis15_ca_vs30_map import Willis15Vs30CA as willis15vs30
#usgs vel model
from query_usgs_v21 import USGSVelModelv21 as velmodel_usgs
#velocity model
from vel_model import VelModelStationary as velmodel_stat
from vel_model import VelModelSpatialVarying as velmodel_svar

#user functions
def ExpKernel(loc_array, omega, ell, delta=0.001):
    
    #number of points
    n_pt = len(loc_array)
    #number of dimensions
    n_dim = loc_array.ndim

    #distance matrix
    dist_mat = np.array([np.norm(loc - loc_array, axis=1) if n_dim > 1 else np.abs(loc - loc_array)
                         for loc in loc_array])
    
    #covariance matrix
    cov_mat = omega**2 * np.exp(-dist_mat/ell)
    cov_mat += np.diag(np.full(n_pt, delta))
    
    #lower Cholesky decomposition
    chol_mat = np.linalg.cholesky(cov_mat)
    
    return cov_mat, chol_mat

def logmean(data):
    
    data = data[data>0]
    
    return np.exp( np.mean( np.log(data) ) )

def logstd(data):

    data = data[data>0]
    
    return np.std( np.log(data) )

#%% Define Variables
### ======================================
#depth increment
dz = .5
#number of spatially-varying realizations 
n_realiz_svar  = 0
#number of along-depth realizations
n_realiz_depth = 5

#vs30 input option
#   flag_vs30 = 1: use profile's vs30 values
#   flag_vs30 = 2: use wills 15 vs30 values
flag_vs30 = 2

#
flag_legend = False

#profile outliers
out_ds_id  = np.array([1, 3])
out_vel_id = np.array([6, 69])

#original profiles
fname_prof_flatfile = '../../Data/vel_profiles_dataset/all_velocity_profles.csv'

#filename for stationary and spatially varying model hyper-parameters
fname_hypparam_model_stat = '../../Data/regression/model_stationary/all_trunc_stan_hyperparameters.csv'
fname_hypparam_model_svar = '../../Data/regression/model_spatially_varying/all_trunc_stan_hyperparameters.csv'
#filename for stationary and spatially varying model parameters
fname_param_model_stat = '../../Data/regression/model_stationary/all_trunc_stan_parameters.csv'
fname_param_model_svar = '../../Data/regression/model_spatially_varying/all_trunc_stan_parameters.csv'
#filename for stationary and spatially varying model along-depth correlation
fname_dcparam_model_stat = '../../Data/regression/along_depth_correlation/model_stationary_semi_variogram_parameters.csv'
fname_dcparam_model_svar = '../../Data/regression/along_depth_correlation/model_spatially_varying_semi_variogram_parameters.csv'

#usgs vel model
dir_usgs   = '/mnt/halcloud_nfs/glavrent/Research/GP_Vel_profiles/Raw_files/vel_model/USGS_SFB_vel_model/'
fname_usgs = 'USGS_SFCVM_v21-1_detailed.h5'

#bay area region
fn_regions = '../../Data/gis/regions_vel_model.shp'

#output directories
dir_out = '../../Data/misc/examp_profs_regions/'
dir_fig = dir_out + 'figures/'


#%% Load Data
### ======================================
#orignal vel proifles
df_vel_prof = pd.read_csv(fname_prof_flatfile)

#read velocity profile hyper-parameter
df_model_stat_hparam = pd.read_csv(fname_hypparam_model_stat, index_col=0)
df_model_svar_hparam = pd.read_csv(fname_hypparam_model_svar, index_col=0) 

#read along-depth correlation parameters
df_model_stat_dcparam = pd.read_csv(fname_dcparam_model_stat, index_col=0)
df_model_svar_dcparam = pd.read_csv(fname_dcparam_model_svar, index_col=0) 

#read velocity profile parameter locations
df_model_stat_param = pd.read_csv(fname_param_model_stat)
df_model_svar_param = pd.read_csv(fname_param_model_svar)

#read regions' geometry
df_regions = gpd.read_file(fn_regions)
df_regions.rename(columns={'id':'RegID','abbrev':'RegAbbrev','name':'RegName'}, 
                  inplace=True)

#%% Processing
### ======================================
#gather profile information
df_prof_info =  df_model_stat_param.loc[:,['DSID','VelID','DSName','VelName',
                                           'Lon','Lat','X','Y',
                                           'Vs30','Z_max']]
df_prof_info.rename(columns={'Z_max':'Zmax'}, inplace=True)


#identify profiles to plot
if len(out_vel_id):
    idx_vel2keep = ~np.any([np.logical_and(df_prof_info.DSID==d_id, df_prof_info.VelID==v_id).values 
                            for d_id, v_id in zip(out_ds_id, out_vel_id)], axis=0)
else:
    idx_vel2keep =  np.full(len(df_prof_info), True, dtype=bool)

#drop outlier profiles
df_prof_info = df_prof_info.loc[idx_vel2keep,:].reset_index(drop=True)


#convert profile info to geopanads
df_prof_info = gpd.GeoDataFrame(df_prof_info,
                                geometry=gpd.points_from_xy(df_prof_info.Lon.values,
                                                            df_prof_info.Lat.values, 
                                                            crs=df_regions.crs))


#remove profiles outside the region
df_prof_info = gpd.sjoin(df_prof_info, df_regions, how="inner", predicate="within")
df_prof_info.reset_index(drop=True, inplace=True)

#re-organize columns
df_prof_info = df_prof_info[['DSID', 'VelID','RegID',
                             'DSName', 'VelName', 'RegName',
                             'Lon', 'Lat', 'X', 'Y', 'geometry',
                             'Zmax', 'Vs30']]

# Evaluate Velocity Models
# -----------------------------
# Initialization
# ---   ---   ---   ---   ---
#initialize statinoary and spatially varying models
model_vel_stat = velmodel_stat(fname_hparam=fname_hypparam_model_stat)
model_vel_svar_cnd = velmodel_svar(fname_hparam=fname_hypparam_model_svar, fname_dBr=fname_param_model_svar)

#initalize usgs model
model_vel_usgs = velmodel_usgs(dir_usgs, fname_usgs )

# Compute Vs30
# -----------------------------
#initiallize wills model
model_w15ca_vs30 = willis15vs30()

#read median vs30
df_prof_info.loc[:,'Vs30Wills15'], _ = model_w15ca_vs30.lookup( df_prof_info[['Lon','Lat']].values )

df_prof_info.loc[:,'Vs30USGS'] = [model_vel_usgs.GetVs30(latlon) 
                                  for latlon in  df_prof_info[['Lat','Lon']].values]

# Evaluate Velocity Profiles
# -----------------------------
#model parameters
#stationary model
model_stat_sig = df_model_stat_hparam.loc['mean','sigma_vel']
model_stat_ell = df_model_stat_dcparam.loc['mean','range']
model_stat_phi = np.sqrt(df_model_stat_dcparam.loc['mean','sill'])
#stationary model
model_svar_sig = df_model_svar_hparam.loc['mean','sigma_vel']
model_svar_ell = df_model_svar_dcparam.loc['mean','range']
model_svar_phi = np.sqrt(df_model_svar_dcparam.loc['mean','sill'])

#initialize vel prof arrays
vprof_emp_all  = list()
vprof_usgs_all = list()
vprof_stat_all = list()
vprof_svar_all = list()
#initalize residual arrays
res_usgs_all = list()
res_stat_all = list()
res_svar_all = list()

#column names for random realizations
cn_stat_realiz_dvar = ['Vs_drlz%i'%(l+1) for l in range(n_realiz_depth)]
cn_svar_realiz_mean = ['Vs_srlz%i'%(j+1) for j in range( max(n_realiz_svar,1) )]
cn_svar_realiz_dvar = [['Vs_srlz%i_drlz%i'%(j+1,l+1) for j in range( max(n_realiz_svar,1) )] for l in range(n_realiz_depth)]
cn_svar_realiz_dvar = sum(cn_svar_realiz_dvar,[])

#iterate over profiles
print('Grenerate profiles ...')
for k, vprof_info in df_prof_info.iterrows():
    print(f'\tvelocity profile: %s, %s (%i of %i) ...'%(vprof_info.DSName,vprof_info.VelName,k+1,len(df_prof_info)))

    #profile information
    #dataset and profile ids
    vropf_regid   = vprof_info.RegID
    vprof_dsid    = vprof_info.DSID
    vprof_velid   = vprof_info.VelID
    #dataset region and profile names
    vprof_regname = vprof_info.RegName
    vprof_dsname  = vprof_info.DSName
    vprof_velname = vprof_info.VelName
    #parametrization
    #vs30 value
    if   flag_vs30 == 1: vprof_vs30 = vprof_info.Vs30
    elif flag_vs30 == 2: vprof_vs30 = vprof_info.Vs30Wills15
    elif flag_vs30 == 3: vprof_vs30 = vprof_info.Vs30USGS
    #adjustment for offshore profiles
    if  vprof_vs30 <= 0: vprof_vs30 = vprof_info.Vs30    
    #profile location
    vprof_latlon = vprof_info[['Lat','Lon']].values
    
    #define filename
    df_prof_info.loc[k,'VelFName'] = '%i-%s_%i-%i_%s_%s'%(vropf_regid, vprof_regname, 
                                                          vprof_dsid, vprof_velid, vprof_dsname, vprof_velname)
        
    #indices stationary and spatially varying profiles
    idx_stat = np.logical_and(df_model_stat_param.DSID==vprof_info.DSID, df_model_stat_param.VelID==vprof_info.VelID)
    idx_svar = np.logical_and(df_model_svar_param.DSID==vprof_info.DSID, df_model_svar_param.VelID==vprof_info.VelID)
    #indices original profile
    i_vprof = np.logical_and(df_vel_prof.DSID==vprof_info.DSID, df_vel_prof.VelID==vprof_info.VelID)

    #avergate time velocity
    Dt = (df_vel_prof.loc[i_vprof,'Depth'].diff() / df_vel_prof.loc[i_vprof,'Vs']).sum()
    Vs_avg = vprof_info.Zmax / Dt
    #store information
    df_prof_info.loc[k,'Dt']     = Dt
    df_prof_info.loc[k,'VsAvg'] = Vs_avg 
    
    #depth array
    z_array = np.arange(0, vprof_info.Zmax+0.01, dz)
    
    #empirical data
    vs_array_emp = interp.interp1d(x=df_vel_prof.loc[i_vprof,'Depth'].values, 
                                   y=df_vel_prof.loc[i_vprof,'Vs'].values,
                                   fill_value=tuple(df_vel_prof.loc[i_vprof,'Vs'].values[[0,-1]].tolist()),
                                   kind='next', bounds_error=False)(z_array)
    
    #usgs model
    # vs_array_usgs = model_vel_usgs.QueryZ(vprof_latlon, z=z_array)[2].Vs.values
    df_vel_usgs = model_vel_usgs.QueryZ(vprof_latlon, z=z_array)[2]
    if len(df_vel_usgs)==0 or np.any(np.isnan(df_vel_usgs.Vs.values)):
        #
        df_vel_usgs = model_vel_usgs.QueryZ(vprof_latlon, z=np.arange(0, vprof_info.Zmax+500.01, dz))[2]      
        #new depth horizion
        z_shift = df_vel_usgs.loc[np.argmax(~np.isnan(df_vel_usgs.Vs.values)),'z2surf']
        #updated usgs model
        df_vel_usgs = model_vel_usgs.QueryZ(vprof_latlon, z=z_array+z_shift)[2]
    #usgs vel model
    vs_array_usgs = df_vel_usgs.Vs.values
    assert(len(vs_array_usgs)>0 and ~np.any(np.isnan(vs_array_usgs)))
    
    #stationaly model
    #create covariance matrix
    cmat_stat, _ = ExpKernel(z_array, omega=model_stat_phi, ell=model_stat_ell)
    cmat_stat += model_stat_sig**2 - model_stat_phi**2
    #stationaly model - mean
    vs_array_stat = model_vel_stat.Vs(vprof_vs30, z_array)[0].flatten()
    #stationaly model - along depth correlation
    var_stat_drzl = np.random.multivariate_normal(np.zeros(z_array.shape), cmat_stat, n_realiz_depth).T
    var_stat_drzl = var_stat_drzl.reshape(-1, 1) if n_realiz_depth == 0 else var_stat_drzl
    vs_array_stat_drzl = vs_array_stat[:,np.newaxis] * np.exp( var_stat_drzl )
    
    #spatially vayring model
    #create covariance matrix
    cmat_svar, _ = ExpKernel(z_array, omega=model_svar_phi, ell=model_svar_ell)
    cmat_svar += model_svar_sig**2 - model_svar_phi**2
    #spatially vayring model - mean
    vs_array_svar = model_vel_svar_cnd.Vs(vprof_vs30, z_array, latlon=vprof_latlon)[0].flatten()
    vs_array_svar_rlz = model_vel_svar_cnd.Vs(vprof_vs30, z_array, latlon=vprof_latlon, n_realiz=n_realiz_svar)[0].squeeze()
    vs_array_svar_rlz = vs_array_svar_rlz.reshape(-1, 1) if n_realiz_svar == 0 else vs_array_svar_rlz
    #spatially vayring model - along depth correlation
    var_svar_drzl = np.random.multivariate_normal(np.zeros(z_array.shape), cmat_svar, n_realiz_depth).T
    var_svar_drzl = var_svar_drzl.reshape(-1, 1) if n_realiz_depth == 0 else var_svar_drzl
    vs_array_svar_rlz_drzl = [vs_array_svar_rlz[:,j][:,np.newaxis] * np.exp( var_svar_drzl ) 
                              for j in range( max(n_realiz_svar,1) )]
    vs_array_svar_rlz_drzl = np.hstack(vs_array_svar_rlz_drzl)

    #summarize velocity proifles (without half-space)
    vprof_emp  = pd.DataFrame({'z':z_array,'Vs':vs_array_emp})
    vprof_usgs = pd.DataFrame({'z':z_array,'Vs':vs_array_usgs})
    vprof_stat = pd.DataFrame({'z':z_array,'Vs':vs_array_stat})
    vprof_svar = pd.DataFrame({'z':z_array,'Vs':vs_array_svar})
    #add random realizations (slope)
    vprof_svar.loc[:,cn_svar_realiz_mean] = vs_array_svar_rlz
    #add random realizations (along-depth)
    vprof_stat.loc[:,cn_stat_realiz_dvar] = vs_array_stat_drzl
    vprof_svar.loc[:,cn_svar_realiz_dvar] = vs_array_svar_rlz_drzl        
    
    #store velocity profiles
    vprof_emp_all.append(vprof_emp)
    vprof_usgs_all.append(vprof_usgs)
    vprof_stat_all.append(vprof_stat)
    vprof_svar_all.append(vprof_svar)
    #residuals
    res_usgs_all.append(np.log(vprof_usgs.Vs) - np.log(vprof_emp.Vs))
    res_stat_all.append(np.log(vprof_stat.Vs) - np.log(vprof_emp.Vs))
    res_svar_all.append(np.log(vprof_svar.Vs) - np.log(vprof_emp.Vs))

# Summarize Regions 
# -----------------------------
print('Summarize regions ...')
for k, reg_info in df_regions.iterrows():
    print(f'\region: %s (%i of %i) ...'%(reg_info.RegName,k+1,len(df_regions)))

    #identify profiles within region
    i_prof_reg = reg_info.RegID == df_prof_info.RegID.values

    #number of profiles
    n_profs = np.sum( i_prof_reg )
    #vs30 statistics 
    vreg_vs30data_mu    = logmean( df_prof_info.loc[i_prof_reg,'Vs30'].values )
    vreg_vs30data_sd    = logstd( df_prof_info.loc[i_prof_reg,'Vs30'].values )
    vreg_vs30wills15_mu = logmean( df_prof_info.loc[i_prof_reg,'Vs30Wills15'].values )
    vreg_vs30wills15_sd = logstd( df_prof_info.loc[i_prof_reg,'Vs30Wills15'].values )
    vreg_vs30usgs_mu    = logmean( df_prof_info.loc[i_prof_reg,'Vs30USGS'].values )
    vreg_vs30usgs_sd    = logstd( df_prof_info.loc[i_prof_reg,'Vs30USGS'].values )
    
    #number of profiles
    df_regions.loc[k,'NoProfs'] = n_profs
    #vs30 summary
    df_regions.loc[k,['Vs30mu','Vs30sd']]               = [vreg_vs30data_mu, vreg_vs30data_sd] 
    df_regions.loc[k,['Vs30Wills15mu','Vs30Wills15sd']] = [vreg_vs30wills15_mu, vreg_vs30wills15_sd]
    df_regions.loc[k,['Vs30USGSmu','Vs30USGSsd']]       = [vreg_vs30usgs_mu, vreg_vs30usgs_sd]


#%% Save Profiles
### ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#summarize regional velocity profiles
fn_reg_info = 'region_information'
df_regions.drop(columns='geometry').to_csv(dir_out + fn_reg_info + '.csv', index=False)

#iterate over profiles
for k, vprof_info in df_prof_info.iterrows():
    #profile information
    fn_vprof_main = vprof_info.VelFName
    
    #save velocity profiles
    vprof_emp_all[k].to_csv(dir_out  + fn_vprof_main + '_emp'  + '.csv', index=False)
    vprof_usgs_all[k].to_csv(dir_out + fn_vprof_main + '_usgs' + '.csv', index=False)
    vprof_stat_all[k].to_csv(dir_out + fn_vprof_main + '_stat' + '.csv', index=False)
    vprof_svar_all[k].to_csv(dir_out + fn_vprof_main + '_svar' + '.csv', index=False)
    

#%% Plotting
### ======================================
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

for k, vprof_info in df_prof_info.iterrows():
    #profile information
    fn_vprof_main = vprof_info.VelFName

    #create figure - with random realizations
    fig, ax = plt.subplots(figsize = (10,10))
    #plot median profiles
    hl_emp  = ax.step(vprof_emp_all[k].Vs,  vprof_emp_all[k].z,  linewidth=3.0, color='black', zorder=3, label='Velocity Profile')
    hl_usgs = ax.plot(vprof_usgs_all[k].Vs, vprof_usgs_all[k].z, linewidth=3.0, color='black', zorder=2, linestyle='dashed', label='USGS SFBA')
    hl_stat = ax.plot(vprof_stat_all[k].Vs, vprof_stat_all[k].z, linewidth=3.0, color='C1', zorder=1, label='Stationary')
    hl_svar = ax.plot(vprof_svar_all[k].Vs, vprof_svar_all[k].z, linewidth=3.0, color='C0', zorder=1, label='Spatially Varying')
    #plot spatially varying slope realizations
    # hl_svar_realiz = ax.plot(vprof_svar_all[k].loc[:,cn_svar_realiz_mean], vprof_svar_all[k].z, linewidth=.5, color='C0', zorder=0, label=f'Spatially Varying Model\n(Uncertainty)')
    #plot stationary with along-depth random realizations
    hl_stat_drlz = ax.plot(vprof_stat_all[k].loc[:,cn_stat_realiz_dvar], vprof_stat_all[k].z, linewidth=0.5, color=hl_stat[0].get_color(), linestyle='dashed', zorder=0, 
                           label=r'Stationary Model\n(Random Realizations)')
    #plot spatially varying wiht along-depth random realizations
    hl_svar_drlz = ax.plot(vprof_svar_all[k].loc[:,cn_svar_realiz_dvar], vprof_stat_all[k].z, linewidth=0.5, color=hl_svar[0].get_color(), linestyle='dashed', zorder=0, 
                           label=r'Spatially Varying Model\n(Random Realizations)')
    #edit properties
    ax.set_xlabel('$V_S$ (m/sec)',  fontsize=30)
    ax.set_ylabel('Depth (m)',      fontsize=30)
    ax.grid(which='both')
    # ax.legend(handles=hl_emp+hl_usgs+hl_stat+hl_svar+[hl_svar_realiz[0]], loc='upper right', fontsize=30)
    if flag_legend: ax.legend(handles=hl_emp+hl_usgs+hl_stat+hl_svar, loc='lower right', fontsize=30)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.set_xlim([0, 2500])
    ax.set_ylim([0, 150])
    ax.invert_yaxis()
    fig.tight_layout()
    #save figure
    fig.savefig( dir_fig + fn_vprof_main + '.png' )
    