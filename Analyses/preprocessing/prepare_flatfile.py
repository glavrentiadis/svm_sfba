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

#read Jian velocity profiles
def ReadJianProfs(dir_name, fname_coor=None, dsid=np.nan, dsname='N/A'):
    
    #directory filenames
    fname_dir = np.array( os.listdir(dir_name) )
    #velocity filenames
    fname_vel = fname_dir[ [bool(re.search('\.txt$',f_d)) for f_d in fname_dir] ]
    n_vel = len(fname_vel)
    
    #velocity profile names
    p = re.compile('profile_(.*)\.txt')
    name_vel = [p.findall(f_v)[0] for f_v in fname_vel]
    
    #read coordinates
    df_coor = pd.read_csv(fname_coor) if not fname_coor is None else None 
    
    #read and merge velocity profiles
    df_vel_profs = []
    for k, (f_v, n_v) in enumerate(zip(fname_vel, name_vel)):
        print(r'reading prof  %i of %i'%(k,n_vel))
        
        #read vel profs
        df_v = pd.read_csv(dir_name + f_v)
        #add prof id and name
        df_v.loc[:,'VelID']   = k
        df_v.loc[:,'VelName'] = n_v
        #add dataset id and name
        df_v.loc[:,'DSID']   = dsid
        df_v.loc[:,'DSName'] = dsname
        
        #compute vs30 
        df_v.loc[:,'Vs30'] = CalcVS30(df_v.Depth.values, df_v.Vs.values)
        
        #compute 
        if not df_coor is None:
            break
        else:
            df_v.loc[:,['lat','lon']] = np.full(2,np.nan)
        
        #reorder columns
        df = df_v[['DSID','DSName','VelID','VelName','Vs30','lat','lon','Depth','Vs']]

        #merge dataframes
        df_vel_profs.append(df)
        
    return pd.concat(df_vel_profs, axis=0).reset_index(drop=True)

#read Boore velocity profiles
def ReadBooreProfs(dir_name, fname_coor=None, dsid=np.nan, dsname='N/A'):

    #directory filenames
    fname_dir = np.array( os.listdir(dir_name) )
    #velocity filenames
    fname_vel = fname_dir[ [bool(re.search('\.txt$',f_d)) for f_d in fname_dir] ]
    n_vel = len(fname_vel)
    
    #velocity profile names
    p = re.compile('(.*)\.txt')
    name_vel = [p.findall(f_v)[0] for f_v in fname_vel]
    
    #read coordinates
    df_coor = pd.read_csv(fname_coor) if not fname_coor is None else None 
    
    #read and merge velocity profiles
    df_vel_profs = []
    for k, (f_v, n_v) in enumerate(zip(fname_vel, name_vel)):
        print(r'reading prof  %i of %i'%(k,n_vel))
        
        #read vel profs
        df_v = pd.read_csv(dir_name + f_v, names=['Depth','Vs'])
        #add prof id and name
        df_v.loc[:,'VelID']   = k
        df_v.loc[:,'VelName'] = n_v
        #add dataset id and name
        df_v.loc[:,'DSID']   = dsid
        df_v.loc[:,'DSName'] = dsname

        #compute vs30 
        df_v.loc[:,'Vs30'] = CalcVS30(df_v.Depth.values, df_v.Vs.values)
        
        #compute 
        if not df_coor is None:
            break
        else:
            df_v.loc[:,['lat','lon']] = np.full(2,np.nan)
        
        #reorder columns
        df = df_v[['DSID','DSName','VelID','VelName','Vs30','lat','lon','Depth','Vs']]
          
        #merge dataframes
        df_vel_profs.append(df)

    return pd.concat(df_vel_profs, axis=0).reset_index(drop=True)

#read USGS velocity profiles
def ReadUSGSProfs(dir_name, fname_coor=None, dsid=np.nan, dsname='N/A'):

    #directory filenames
    fname_dir = np.array( os.listdir(dir_name) )
    #velocity filenames
    fname_vel = fname_dir[ [bool(re.search('\.csv$',f_d)) for f_d in fname_dir] ]
    n_vel = len(fname_vel)
    
    #velocity profile names
    p = re.compile('(.*)_vel_profile.csv')
    name_vel = [p.findall(f_v)[0] for f_v in fname_vel]
    
    #read coordinates
    df_coor = pd.read_csv(fname_coor) if not fname_coor is None else None 
    
    #read and merge velocity profiles
    df_vel_profs = []
    for k, (f_v, n_v) in enumerate(zip(fname_vel, name_vel)):
        print(r'reading prof  %i of %i (%s)'%(k,n_vel,f_v))
        
        #read vel profs
        df_v = pd.read_csv(dir_name + f_v + '/profile.txt', names=['Depth','Vs'])
        #add prof id and name
        df_v.loc[:,'VelID']   = k
        df_v.loc[:,'VelName'] = n_v
        #add dataset id and name
        df_v.loc[:,'DSID']   = dsid
        df_v.loc[:,'DSName'] = dsname

        #compute vs30 
        df_v.loc[:,'Vs30'] = CalcVS30(df_v.Depth.values, df_v.Vs.values)
        
        #compute 
        if not df_coor is None:
            break
        else:
            df_v.loc[:,['lat','lon']] = np.full(2,np.nan)
        
        #reorder columns
        df = df_v[['DSID','DSName','VelID','VelName','Vs30','lat','lon','Depth','Vs']]
          
        #merge dataframes
        df_vel_profs.append(df)

    return pd.concat(df_vel_profs, axis=0).reset_index(drop=True)

#read VSPDB velocity profiles
def ReadVSPDBProfs(dir_name, fname_coor=None, dsid=np.nan, dsname='N/A'):

    #directory filenames
    fname_dir = np.array( os.listdir(dir_name) )
    #velocity filenames
    fname_vel = fname_dir[ [bool(re.search('\.txt$',f_d)) for f_d in fname_dir] ]
    n_vel = len(fname_vel)
    
    #velocity profile names
    p = re.compile('(.*)_velocityProfile.*.txt')
    name_vel = [p.findall(f_v)[0] for f_v in fname_vel]
    
    #read coordinates
    df_coor = pd.read_csv(fname_coor) if not fname_coor is None else None 
    
    #read and merge velocity profiles
    df_vel_profs = []
    for k, (f_v, n_v) in enumerate(zip(fname_vel, name_vel)):
        print(r'reading prof  %i of %i'%(k,n_vel))
        
        #read vel profs
        df_v = pd.read_csv(dir_name + f_v)
        #add prof id and name
        df_v.loc[:,'VelID']   = k
        df_v.loc[:,'VelName'] = n_v
        #add dataset id and name
        df_v.loc[:,'DSID']   = dsid
        df_v.loc[:,'DSName'] = dsname

        #compute vs30 
        df_v.loc[:,'Vs30'] = CalcVS30(df_v.Depth.values, df_v.Vs.values)
        
        #compute 
        if not df_coor is None:
            break
        else:
            df_v.loc[:,['lat','lon']] = np.full(2,np.nan)
        
        #reorder columns
        df = df_v[['DSID','DSName','VelID','VelName','Vs30','lat','lon','Depth','Vs']]
          
        #merge dataframes
        df_vel_profs.append(df)

    return pd.concat(df_vel_profs, axis=0).reset_index(drop=True)


#%% Define Variables
### ======================================

#Jian profiles
dir_velprofs_Jian     = '../Raw_files/Datasets/csv_Jian/'
fname_coor_Jian       = None
#Boore profiles
dir_velprofs_Boore    = '../Raw_files/Datasets/Boore/'
fname_coor_Boore      = None
#USGS profiles (stations)
dir_velprofs_USGSsta  = '../Raw_files/Datasets/USGS/Stations/'
fname_coor_USGSsta    = None
#USGS profiles (mesh)
dir_velprofs_USGSmesh = '../Raw_files/Datasets/USGS/Mesh/'
fname_coor_USGSmesh   = None
#VSPDB_Vs_Profiles
dir_velprofs_VSPDB    = '../Raw_files/Datasets/VSPDB_Vs_Profiles/Vs/txt/'
fname_coor_VSPDB      = None

#output directory
dir_out = '../Data/vel_profiles/'

#%% Load Data
### ======================================
df_vel_profs = []

#read Jian profiles
print("Reading Jian's profiles")
df_vel_profs.append( ReadJianProfs(dir_name=dir_velprofs_Jian,     fname_coor=fname_coor_Jian,     dsid=1, dsname='Jian') )
#read Boore profiles
print("Reading Boore's profiles")
df_vel_profs.append( ReadBooreProfs(dir_name=dir_velprofs_Boore,   fname_coor=fname_coor_Boore,    dsid=2, dsname='Boore') )
#read USGS profiles (stations)
print("Reading USGS profiles (stations)")
df_vel_profs.append( ReadUSGSProfs(dir_name=dir_velprofs_USGSsta,  fname_coor=fname_coor_USGSsta,  dsid=3, dsname='USGS-stations') )
#read USGS profiles (mesh)
print("Reading USGS profiles (mesh)")
df_vel_profs.append( ReadUSGSProfs(dir_name=dir_velprofs_USGSmesh, fname_coor=fname_coor_USGSmesh, dsid=4, dsname='USGS-mesh') )
#read VSPDB profiles
print("Reading VSPDB profiles")
df_vel_profs.append( ReadVSPDBProfs(dir_name=dir_velprofs_VSPDB,   fname_coor=fname_coor_VSPDB,    dsid=5, dsname='VSPDB') )


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
#save USGSsta profiles
df_vel_profs.loc[df_vel_profs.DSID==3,].to_csv( dir_out+ 'USGSsta_velocity_profles.csv',  index=False )
#save USGSmesh profiles
df_vel_profs.loc[df_vel_profs.DSID==4,].to_csv( dir_out+ 'USGSmesh_velocity_profles.csv', index=False )
#save VSPDB profiles
df_vel_profs.loc[df_vel_profs.DSID==5,].to_csv( dir_out+ 'VSPDB_velocity_profles.csv',    index=False )




