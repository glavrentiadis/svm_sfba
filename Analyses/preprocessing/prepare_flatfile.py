#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:38:15 2022

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

#user functions
#---------------------
#compute vs30
def CalcVS30(depth, vel):
    
    #interpolate depths at 1m interval
    d_array  = np.arange(0,31)
    dz_array = np.diff(d_array)
    vs_array = interp.interp1d(x=depth,y=vel,kind='previous', bounds_error=False, fill_value='extrapolate')(d_array[:-1])

    #compute Vs30
    t_array = dz_array/vs_array
    vs30 = 30/np.sum(t_array)
    
    return vs30

#read general velocity profiles
def ReadGenProfs(dir_name, fname_vprof_info, flag_header = True, dsid=np.nan, filter_vname='(.*)', dsname='N/A'):
    
    #read vel profile info
    df_vprof_info = pd.read_excel(fname_vprof_info)
    n_vel = len(df_vprof_info )
    
    #velocity profile names
    p = re.compile(filter_vname)

    df_v_profs = list()
    for k, (_, df_vp_info) in enumerate(df_vprof_info.iterrows()):
        print(r'reading prof  %i of %i (%s)'%(k,n_vel,df_vp_info.dataset))
        
        #file name
        f_v = df_vp_info.dataset

        #read vel profs
        df_v = pd.read_csv(dir_name+f_v) if flag_header else pd.read_csv(dir_name+f_v, names=('Depth','Vs'))

        #velocity profile name
        n_v = p.findall(f_v)[0] 
        if type(n_v) is tuple:  n_v = " ".join(n_v)

        #top of layer depth
        z  = df_v.Depth.values
        #Vs values
        vs = df_v.Vs.values
        #skip profiles with nan
        if np.any(np.isnan(vs)):
            print(f'\tSkipping prof %s ...'%(df_vp_info.dataset))
            continue
        #assert top layer depth reported
        assert(np.abs(z[0])<1e-9),'Error. Depths to top of layers should be reported.'

        #add prof id and name
        df_v.loc[:,'VelID']   = k
        df_v.loc[:,'VelName'] = n_v
        #add dataset id and name
        df_v.loc[:,'DSID']   = dsid
        df_v.loc[:,'DSName'] = dsname
        
        #flag Z1.0
        df_v.loc[:,'flag_Z1'] = np.cumsum( vs >= 1000 ) >= 1
        
        #compute layer thickness and mid-point depth
        df_v.loc[:,'Thk'] = np.append(np.diff(z), 0)
        #midpoint depth
        z_mp = np.convolve(z, np.ones(2), "valid")/2
        df_v.loc[:,'Depth_MPt'] = np.append(z_mp, 0)
        
        #compute vs30 
        df_v.loc[:,'Vs30'] = CalcVS30(df_v.Depth.values, df_v.Vs.values)
        
        #copy coordinates 
        df_v.loc[:,['Lat','Lon']] = df_vp_info[['lat','long']].values
        
        #reorder columns
        df_v = df_v[['DSID','DSName','VelID','VelName','Vs30','Lat','Lon','Depth','Depth_MPt','Thk','Vs','flag_Z1']]

        #merge dataframes
        df_v_profs.append(df_v)
        
    return pd.concat(df_v_profs, axis=0).reset_index(drop=True)


#%% Define Variables
### ======================================

#Jian profiles
dir_velprofs_Jian     = '../../Raw_files/Datasets_20220915/Jian/dataset/'
fname_vprof_info_Jian = '../../Raw_files/Datasets_20220915/Jian/Jian.xlsx'
filter_vname_Jian     = 'profile_(.*)\.txt'
#Boore profiles
dir_velprofs_Boore     = '../../Raw_files/Datasets_20220915/Boore/dataset/'
fname_vprof_info_Boore = '../../Raw_files/Datasets_20220915/Boore/Boore.xlsx'
filter_vname_Boore     = '(.*)\.txt'
#VSPDB_Vs_Profiles
dir_velprofs_VSPDB     = '../../Raw_files/Datasets_20220915/VSPDB/dataset/'
fname_vprof_info_VSPDB = '../../Raw_files/Datasets_20220915/VSPDB/VSPDB.xlsx'
filter_vname_VSPDB     = '(.*)_velocityProfile_(.*)\.txt'

#output directory
dir_out = '../../Data/vel_profiles/'

#%% Load Data
### ======================================
df_vel_profs = []

#read Jian profiles
print("Reading Jian's profiles")
df_vel_profs.append( ReadGenProfs(dir_name=dir_velprofs_Jian,  fname_vprof_info=fname_vprof_info_Jian,
                                  dsid=1, dsname='Jian', filter_vname=filter_vname_Jian) )
#read Boore profiles
print("Reading Boore's profiles")
df_vel_profs.append( ReadGenProfs(dir_name=dir_velprofs_Boore, fname_vprof_info=fname_vprof_info_Boore,
                                  dsid=2, dsname='Boore', filter_vname=filter_vname_Boore, flag_header=False) )
#read VSPDB profiles
print("Reading VSPDB profiles")
df_vel_profs.append( ReadGenProfs(dir_name=dir_velprofs_VSPDB, fname_vprof_info=fname_vprof_info_VSPDB, 
                                  dsid=3, dsname='VSPDB', filter_vname=filter_vname_VSPDB) )

#concatinate all data
df_vel_profs = pd.concat(df_vel_profs, axis=0).reset_index(drop=True)

#%% Save Data
### ======================================
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#save all profiles
df_vel_profs.to_csv( dir_out+'all_velocity_profles.csv', index=False )
#save Jian profiles
df_vel_profs.loc[df_vel_profs.DSID==1,].to_csv( dir_out+ 'Jian_velocity_profles.csv',     index=False )
#save Boore profiles
df_vel_profs.loc[df_vel_profs.DSID==2,].to_csv( dir_out+ 'Boore_velocity_profles.csv',    index=False )
#save VSPDB profiles
df_vel_profs.loc[df_vel_profs.DSID==3,].to_csv( dir_out+ 'VSPDB_velocity_profles.csv',    index=False )




