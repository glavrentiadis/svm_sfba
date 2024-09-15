#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 14:49:02 2023

@author: glavrent
"""
import numpy as np

def movingmean(y_array, x_array, x_bin):
    '''
    Moving Mean Statistics

    Parameters
    ----------
    y_array : np.array
        Response variable.
    x_array : np.array()
        Conditional variable.
    x_bin : np.array()
        Conditional variable bins.

    Returns
    -------
    x_mid : np.array()
        Mid-point of conditional variable bins.
    y_mmed : np.array()
        Moving median.
    y_mmean : np.array()
        Moving mean.
    y_mstd : np.array()
        Moving standard deviation.
    y_m16prc : np.array()
        Moving 16th percentile.
    y_m84prc : np.array()
        Moving 84th percentile.
    '''
    
    #bins' mid point
    x_mid = np.array([(x_bin[j]+x_bin[j+1])/2  for j in range(len(x_bin)-1)])
    
    #binned residuals
    y_mmed   = np.full(len(x_mid), np.nan)
    y_mmean  = np.full(len(x_mid), np.nan)
    y_mstd   = np.full(len(x_mid), np.nan)
    y_m16prc = np.full(len(x_mid), np.nan)
    y_m84prc = np.full(len(x_mid), np.nan)
    
    #iterate over residual bins
    for k in range(len(x_mid)):
        #binned residuals
        i_bin = np.logical_and(x_array >= x_bin[k], x_array < x_bin[k+1])
        #summarize statistics
        y_mmed[k]   = np.median(   y_array[i_bin] )
        y_mmean[k]  = np.mean(     y_array[i_bin])
        y_mstd[k]   = np.std(      y_array[i_bin])
        y_m16prc[k] = np.quantile( y_array[i_bin], 0.16) if i_bin.sum() else np.nan 
        y_m84prc[k] = np.quantile( y_array[i_bin], 0.84) if i_bin.sum() else np.nan
        
    return x_mid, y_mmed, y_mmean, y_mstd, y_m16prc, y_m84prc
