#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 12:50:59 2023

@author: glavrent
"""

#load libraries
import numpy  as np
#ploting libraries
import matplotlib.pyplot as plt
#velocity model
from vel_model import VelModelSpatialVarying


#initialize object
fn_dBr_coeffs = '../../Data/regression/model_spatially_varying/all_trunc_stan_parameters.csv'
ba_vmodel = VelModelSpatialVarying(fname_dBr=fn_dBr_coeffs)


#single profile
#---   ---   ---   ---   ---
#time average shear wave velocity
vs30 = 300
#site coordinates
site_latlon = [37.871960, -122.259094]
site_latlon = [37.53,	-122.26]

#depth array
z_array = np.arange(250.1)

#vel profile
vel_med, vel_glob, vel_sig = ba_vmodel.Vs(vs30, z_array, latlon=site_latlon)

#plot vel profile
fig, ax = plt.subplots(figsize=(10,15))
hl1 = ax.plot(vel_med, z_array,  linestyle='solid',  drawstyle='default', linewidth=3, label='$V_{S30}=%.i$'%vs30)
hl2 = ax.plot(vel_glob, z_array, linestyle='dashed', drawstyle='default', linewidth=3, label='$V_{S30}=%.i$ (global)'%vs30, color=hl1[0].get_color())
ax.set_xlabel('Shear-wave Velocity (m/sec)', fontsize=26)
ax.set_ylabel('Depth (m)',    fontsize=26)
ax.legend(fontsize=26, loc='lower left')
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_ylim([z_array.min(), z_array.max()])
ax.invert_yaxis()
ax.grid()
fig.tight_layout()


#multiple profile
#---   ---   ---   ---   ---
#time average shear wave velocity
vs30_array  = [300, 500, 800]
site_latlon = [[37.53, -122.26], [37.82, -122.36], [37.45,	-122.25]]

#depth array
z_array = np.arange(250.1)

#vel profile
vel_med, vel_glob, vel_sig = ba_vmodel.Vs(vs30_array, z_array, latlon=site_latlon)

#plot vel profile
fig, ax = plt.subplots(figsize=(10,15))
hl1 = ax.plot(vel_med,  z_array, linestyle='solid', drawstyle='default', linewidth=3)
for h, v in zip(hl1, vel_glob.T): ax.plot(v, z_array, linestyle='dashed', drawstyle='default', linewidth=3, color=h.get_color())
ax.set_xlabel('Shear-wave Velocity (m/sec)', fontsize=26)
ax.set_ylabel('Depth (m)',    fontsize=26)
ax.legend([r'$V_{S30}=%.i m/sec$'%vs30 for vs30 in vs30_array], 
          fontsize=26, loc='lower left')
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_ylim([z_array.min(), z_array.max()])
ax.invert_yaxis()
ax.grid()
fig.tight_layout()
