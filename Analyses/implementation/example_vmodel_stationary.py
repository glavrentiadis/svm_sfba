#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:24:36 2023

@author: glavrent
"""

#load libraries
import numpy  as np
#ploting libraries
import matplotlib.pyplot as plt
#velocity model
from vel_model import VelModelStationary


#initialize object
ba_vmodel = VelModelStationary()


#single profile
#---   ---   ---   ---   ---
#time average shear wave velocity
vs30 = 300
#depth array
z_array = np.arange(250.1)

#vel profile
vel_med, vel_sig = ba_vmodel.Vs(vs30, z_array)

#plot vel prof
fig, ax = plt.subplots(figsize=(10,15))
ax.plot(vel_med, z_array, linestyle='solid', drawstyle='default', linewidth=2, label='$V_{S30}=%.i$'%vs30)
ax.set_xlabel('Shear-wave Velocity (m/sec)', fontsize=26)
ax.set_ylabel('Depth (m)',    fontsize=26)
ax.legend(fontsize=26, loc='lower left')
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.invert_yaxis()
ax.grid()
fig.tight_layout()


#multiple profile
#---   ---   ---   ---   ---
#time average shear wave velocity
vs30_array = [150, 300, 500, 800]
#depth array
z_array = np.arange(250.1)

#vel profile
vel_med, vel_sig = ba_vmodel.Vs(vs30_array, z_array)

#plot vel prof
fig, ax = plt.subplots(figsize=(10,15))
ax.plot(vel_med, z_array, linestyle='solid', drawstyle='default', linewidth=2)
ax.set_xlabel('Shear-wave Velocity (m/sec)', fontsize=26)
ax.set_ylabel('Depth (m)',    fontsize=26)
ax.legend([r'$V_{S30}=%.i m/sec$'%vs30 for vs30 in vs30_array], 
          fontsize=26, loc='lower left')
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.invert_yaxis()
ax.grid()
fig.tight_layout()



