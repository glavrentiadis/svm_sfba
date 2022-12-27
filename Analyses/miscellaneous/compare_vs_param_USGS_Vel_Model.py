#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:34:55 2022

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
sys.path.insert(0,'../python_lib/usgs')
from query_usgs import USGSVelModel


# %% Define Input Variables
# ======================================
#input/output flag
flag_io = 1

# input files
if   flag_io == 1: fname_velinfo = '../../Data/vel_profiles_dataset/all_velocity_profles_info.csv'
elif flag_io == 2: fname_velinfo = '../../Data/vel_profiles_dataset/Boore_velocity_profles_info.csv'
elif flag_io == 3: fname_velinfo = '../../Data/vel_profiles_dataset/Jian_velocity_profles_info.csv'
elif flag_io == 4: fname_velinfo = '../../Data/vel_profiles_dataset/VSPDB_velocity_profles_info.csv'

# output directory
dir_out = '../../Data/misc/USGS_comparison'
# output filename
if   flag_io == 1: fname_out = 'all_velocity_profles_info_USGS'
elif flag_io == 2: fname_out = 'Boore_velocity_profles_info_USGS'
elif flag_io == 3: fname_out = 'Jian_velocity_profles_info_USGS'
elif flag_io == 4: fname_out = 'VSPDB_velocity_profles_info_USGS'

#USGS
dir_velmodel = '/mnt/halcloud_nfs/glavrent/Research/Other_projects/Query_USGS/vel_model/'
fname_velmodel     = 'USGSBayAreaVM-08.3.0.etree'
fname_ext_velmodel = 'USGSBayAreaVMExt-08.3.0.etree'


#%% Load Data
### ======================================
#read vel prof info
df_velinfo = pd.read_csv(fname_velinfo)
n_v = len(df_velinfo)

#initialize extended vel model
usgs_vel_model = USGSVelModel(dir_velmodel,fname_velmodel, fname_ext_velmodel)


#%% Process Data
### ======================================
#Compute Vs30, Z1.0 and Z2.5
#----  ----  ----  ----  ----  
#iterate over different locations
print('Reading Vel Profiles:')
for k, v_info in df_velinfo.iterrows():
    print('\tprocessing %s %10.s (%i of %i) ...'%(v_info.DSName, v_info.VelName, k+1, n_v))
    #coordinates of i^th point
    latlon_i = v_info[['Lat','Lon']]
    
    if not usgs_vel_model.Query(latlon_i)[2].empty:
        #compute vs30, Z1.0 and Z2.5
        df_velinfo.loc[k,'usgs_Vs30'] = usgs_vel_model.GetVs30(latlon_i)
        df_velinfo.loc[k,'usgs_Z1.0'] = usgs_vel_model.GetZ1_0(latlon_i) 
        df_velinfo.loc[k,'usgs_Z2.5'] = usgs_vel_model.GetZ2_5(latlon_i)
print('Done!')

# %% Output
# ======================================
#create output directories
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

#save vel info dataframe
df_velinfo.to_csv(dir_out+fname_out+'.csv', index=False)
