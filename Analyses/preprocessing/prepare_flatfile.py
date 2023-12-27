#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:38:15 2022

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
#geographic coordinates
import pyproj

#user functions
#---------------------
#compute vs30
def CalcVS30(depth, vel):
    
    #interpolate depths at 1m interval
    d_array  = np.arange(0,30.05,0.05)
    dz_array = np.diff(d_array)
    vs_array = interp.interp1d(x=depth,y=vel,kind='next', bounds_error=False, fill_value=(vel[0],vel[-1]))(d_array[:-1])

    #compute Vs30
    t_array = dz_array/vs_array
    vs30 = 30/np.sum(t_array)
    
    return vs30

#read general velocity profiles
def ReadGenProfs(dir_name, fname_vprof_info, flag_header=True, sep=',', dsid=np.nan, filter_vname='(.*)', dsname='N/A'):
    
    #read vel profile info
    df_vprof_info = pd.read_excel(fname_vprof_info)
    n_vel = len(df_vprof_info )
    
    #velocity profile names
    p = re.compile(filter_vname)

    #iterate over vel profiles
    df_v_profs = list()
    for k, (_, df_vp_info) in enumerate(df_vprof_info.iterrows()):
        print(r'reading prof  %i of %i (%s)'%(k+1,n_vel,df_vp_info.dataset))
        
        #file name
        f_v = df_vp_info.dataset

        #read vel profs
        df_v = pd.read_csv(dir_name+f_v, sep=sep) if flag_header else pd.read_csv(dir_name+f_v, names=('Depth','Vs'), sep=sep)

        #velocity profile name
        n_v = p.findall(f_v)[0] 
        if type(n_v) is tuple:  n_v = " ".join(n_v)

        #layer depths
        z  = df_v.Depth.values
        #Vs values
        vs = df_v.Vs.values
        #skip profiles with nan
        if np.any(np.isnan(vs)):
            print(f'\tSkipping prof due to unavailable values: %s ...'%(df_vp_info.dataset))
            continue
        #assert top layer depth reported
        assert(np.abs(z[0])<1e-9),'Error. Top of profile should be reported.'
        #assert(np.abs(vs[0]-vs[1])<1e-9),'Error. Inconsistent format. Assumed step function profile.'
        if not np.abs(vs[0]-vs[1])<1e-9:
            print(f'\tSkipping prof due to inconsistent format: %s ...'%(df_vp_info.dataset))
            continue
        
        #add prof id and name
        df_v.loc[:,'VelID']   = k+1
        df_v.loc[:,'VelName'] = n_v
        #add dataset id and name
        df_v.loc[:,'DSID']   = dsid
        df_v.loc[:,'DSName'] = dsname
        
        #max depth
        df_v.loc[:,'Z_max'] = max(z)
        
        #flag Z1.0
        df_v.loc[:,'flag_Z1'] = np.cumsum( vs >= 1000 ) >= 1
        
        #compute layer thickness and mid-point depth
        df_v.loc[:,'Thk'] = np.append(np.nan, np.diff(z))
        #midpoint depth
        z_mp = np.convolve(z, np.ones(2), "valid")/2
        df_v.loc[:,'Depth_MPt'] = np.append(np.nan, z_mp)
        #compute vs30 
        df_v.loc[:,'Vs30'] = CalcVS30(df_v.Depth.values, df_v.Vs.values)
        
        #copy coordinates 
        df_v.loc[:,['Lat','Lon']] = df_vp_info[['lat','long']].values
        
        #reorder columns
        df_v = df_v[['DSID','DSName','VelID','VelName','Vs30','Z_max','Lat','Lon','Depth','Depth_MPt','Thk','Vs','flag_Z1']]

        #merge dataframes
        df_v_profs.append(df_v)
        
    return pd.concat(df_v_profs, axis=0).reset_index(drop=True)

#%% Define Variables
### ======================================
# projection system
utm_zone = '10S'

# Dataset Information
# ---   ---   ---   ---
#Jian profiles
dir_velprofs_SA18     = '../../Raw_files/datasets_20230930/Shi_Asimaki/dataset/'
fname_vprof_info_SA18 = '../../Raw_files/datasets_20230930/Shi_Asimaki_2018.xlsx'
filter_vname_SA18     = 'profile_(.*)\.txt'
#VSPDB_Vs_Profiles
dir_velprofs_VSPDB     = '../../Raw_files/datasets_20230930/VSPDB/dataset/'
fname_vprof_info_VSPDB = '../../Raw_files/datasets_20230930/VSPDB.xlsx'
filter_vname_VSPDB     = '(.*)_velocityProfile_(.*)\.txt'

#outlier profile ids
flag_outlier_rm = True
outlier_ds_id  = [3,  3,  3]
outlier_vel_id = [56, 57, 58]
#outlier_ds_id  = [3,  3,  3,  3,  3,  3]
#outlier_vel_id = [56, 57, 43, 45, 58, 31]

#output directory
# dir_out = '../../Data/vel_profiles/'
dir_out = '../../Data/vel_profiles_dataset/'

#%% Load Data
### ======================================
# load velocity profiles
# ----   ----   ----   ----   ----
df_vel_profs = []

#read Jian profiles
print("Reading Shi and Asimaki 2018 profiles")
df_vel_profs.append( ReadGenProfs(dir_name=dir_velprofs_SA18,  fname_vprof_info=fname_vprof_info_SA18,
                                  dsid=1, dsname='Shi_Asimaki_2018', filter_vname=filter_vname_SA18,
                                  flag_header=False, sep='\t') )

#read VSPDB profiles
print("Reading VSPDB profiles")
df_vel_profs.append( ReadGenProfs(dir_name=dir_velprofs_VSPDB, fname_vprof_info=fname_vprof_info_VSPDB, 
                                  dsid=3, dsname='VSPDB', filter_vname=filter_vname_VSPDB,
                                  flag_header=False, sep='\t') )

#concatinate all data
df_vel_profs = pd.concat(df_vel_profs, axis=0).reset_index(drop=True)

#create velocity profile info
_, vel_idx, vel_inv = np.unique(df_vel_profs[['DSID','VelID']].values, axis=0, return_index=True, return_inverse=True)
n_vel               = vel_idx.shape[0]
df_profs_info       = df_vel_profs[['DSID','DSName','VelID','VelName','Vs30','Z_max','Lat','Lon']].iloc[vel_idx,:].reset_index(drop=True)

# projection system
# ----   ----   ----   ----   ----
#projection system
utmProj = pyproj.Proj("+proj=utm +zone="+utm_zone[:2]+" +ellps=WGS84 +datum=WGS84 +units=km +no_defs")

#compute utm coordinates
v_X  = np.array([utmProj(v.Lon, v.Lat)   for _, v in df_profs_info.iterrows()])  

#store utm coordinates
df_profs_info.loc[:,['X','Y']] = v_X
df_profs_info.loc[:,'UTMzone'] = utm_zone
df_vel_profs.loc[:,['X','Y']] = v_X[vel_inv,:]
df_vel_profs.loc[:,'UTMzone'] = utm_zone


#%% Save Data
### ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#save all profiles
df_vel_profs.to_csv( dir_out+'all_velocity_profles.csv', index=False )
#save Jian profiles
if np.any(df_vel_profs.DSID==1): df_vel_profs.loc[df_vel_profs.DSID==1,].to_csv( dir_out+ 'Shi_Asimaki_velocity_profles.csv',     index=False )
#save Boore profiles
if np.any(df_vel_profs.DSID==2): df_vel_profs.loc[df_vel_profs.DSID==2,].to_csv( dir_out+ 'Boore_velocity_profles.csv',    index=False )
#save VSPDB profiles
if np.any(df_vel_profs.DSID==3): df_vel_profs.loc[df_vel_profs.DSID==3,].to_csv( dir_out+ 'VSPDB_velocity_profles.csv',    index=False )
#save Jian and VSPDB profiles
if np.any(np.isin(df_vel_profs.DSID,[1,3])): df_vel_profs.loc[np.isin(df_vel_profs.DSID,[1,3]),].to_csv( dir_out+ 'Shi_Asimaki_VSPDB_velocity_profles.csv', index=False )

#save velocity profile info
df_profs_info.to_csv( dir_out+'all_velocity_profles_info.csv', index=False )

#save velocity profile info (without outliers)
i_out = np.logical_and(np.isin(df_profs_info.DSID,  outlier_ds_id), np.isin(df_profs_info.VelID, outlier_vel_id))
i_out = np.logical_or(i_out, df_profs_info.Vs30 < 100)
df_profs_info.loc[~i_out,:].to_csv( dir_out+'all_velocity_profles_info_valid.csv', index=False )

