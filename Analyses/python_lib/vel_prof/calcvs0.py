#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:00:06 2023

@author: glavrent
"""
import numpy as np

def calcvs0(vs30, k, n, z_star=2.5):
    '''
    Compute surface shear-wave velocity (Vs0) 

    Parameters
    ----------
    vs30 : float
        Time-average shear-wave velocity at the top 30m.
    k : float
        Shear-wave velocity scale.
    n : float
        Shear-wave velocity exponent.
    z_star : TYPE, optional
        Constant shear-wave velocity depth. The default is 2.5.

    Returns
    -------
    vs0 : float 
        Surface shear-wave velocity.

    '''
    #compute a
    a =-1/n
    #compute Vs0
    vs0 = (z_star + 1/k * np.log(1+k*(30-z_star))) if np.abs(n-1) < 1e-9 else (k*(a+1)*z_star + (1+k*(30-z_star))**(a+1) - 1) / ((a+1)*k)
    vs0 *= vs30/30
    
    return vs0
