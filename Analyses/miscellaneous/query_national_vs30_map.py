#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 19:01:47 2021

@author: glavrent
"""
#load libraries
import pathlib
import numpy as np
import rasterio


class USNationalVs30:
    # USNationalVs30 query US National Vs30 maps from Geyin and Maurer 2006
    #
    # Example:
    #   vs30model = USNationalVs30()
    #   vs30model.lookup( [(-122.258, 37.875)] ) 
    
    def __init__(self, fname_vs30map_m1=None, fname_vs30map_m1alt=None, 
                       fname_vs30map_m2=None, fname_vs30map_m2alt=None):
        #file path
        # dir_data = pathlib.Path(__file__).parent
        dir_data = '/mnt/halcloud_nfs/glavrent/Research/GP_Vel_profiles/Raw_files/vel_model/National_map_Vs30/'
        #vs30 data filenames
        fname_vs30_model1    = fname_vs30map_m1    if not (fname_vs30map_m1    is None) else dir_data+'Model_1.tif'
        fname_vs30_model1alt = fname_vs30map_m1alt if not (fname_vs30map_m1alt is None) else dir_data+'Model_1alt.tif'
        fname_vs30_model2    = fname_vs30map_m2    if not (fname_vs30map_m2    is None) else dir_data+'Model_1.tif'
        fname_vs30_model2alt = fname_vs30map_m2alt if not (fname_vs30map_m2alt is None) else dir_data+'Model_2alt.tif'
        #load vs30 data
        self.vs30map_m1    = rasterio.open( fname_vs30_model1 )
        self.vs30map_m1alt = rasterio.open( fname_vs30_model1alt )
        self.vs30map_m2    = rasterio.open( fname_vs30_model2 )
        self.vs30map_m2alt = rasterio.open( fname_vs30_model2alt )
    
    
    def lookup(self, lonlats):
        return (
            np.fromiter(self.vs30map_m1.sample(lonlats,    1), np.float),
            np.fromiter(self.vs30map_m1alt.sample(lonlats, 1), np.float),
            np.fromiter(self.vs30map_m2.sample(lonlats,    1), np.float),
            np.fromiter(self.vs30map_m2alt.sample(lonlats, 1), np.float)
        )
