#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:53:15 2023

@author: hsalmanitehrani
"""

#pwd should be '/home/hsalmanitehrani/geomodelgrids/geomodelgrids-1.0.0rc2-Linux_x86_64'

import subprocess
import pandas as pd
import numpy as np
import os

data=pd.read_excel("/home/hsalmanitehrani/Downloads/cross_section_1.xlsx", names=['Ind', 'Lon', 'Lat'])
data["Lat"] = data["Lat"].astype('float')
data["Lon"] = data["Lon"].astype('float')

# =============================================================================
# data=pd.read_csv("/home/hsalmanitehrani/Downloads/input_RedBox.csv", dtype='str', header=None,names=['Lon', 'Lat'])
# data["Lat"] = data["Lat"].astype('float')
# data["Lon"] = data["Lon"].astype('float')
# =============================================================================

os.chdir('/home/hsalmanitehrani/geomodelgrids/geomodelgrids-1.0.0rc2-Linux_x86_64')

no_data=[]
for i in range(len(data)):
    a=subprocess.run('source /home/hsalmanitehrani/geomodelgrids/geomodelgrids-1.0.0rc2-Linux_x86_64/setup.sh; geomodelgrids_borehole --models=/home/hsalmanitehrani/geomodelgrids/geomodelgrids-1.0.0rc2-Linux_x86_64/src/tests/data/USGS.h5  --location=%f,%f --max-depth=1000 --dz=1 --output=./cross_section1/%s.out --values=Vs' %(data.Lat[i],data.Lon[i],str(data.Ind[i])),
                   shell=True, executable='/bin/bash', universal_newlines=True)
    if a.returncode!=0:
        no_data.append([i, str([i])])
