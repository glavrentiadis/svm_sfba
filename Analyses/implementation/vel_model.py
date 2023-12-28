#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:24:43 2023

@author: glavrent
"""

#load libraries
import numpy  as np
from numpy import matlib as npmat
from scipy import linalg as scipylinalg
import pandas as pd
#geographic coordinates                         `c
import pyproj

class VelModel:
    def __init__(self, fname_hparam=None, fname_dBr=None, z_star=2.5):
        
        #intialize scaling terms
        self.z_star = z_star
        
        #read model hyper-parameters
        if not fname_hparam is None:
            #read coefficient flatifle
            df_hparam = pd.read_csv(fname_hparam)
            #median scaling
            self.logVs30mid = df_hparam.loc['prc0.50','logVs30mid']
            self.logVs30scl = df_hparam.loc['prc0.50','logVs30scl']
            self.r1         = df_hparam.loc['prc0.50','r1']
            self.r2         = df_hparam.loc['prc0.50','r2']
            self.r3         = df_hparam.loc['prc0.50','r3']
            self.s2         = df_hparam.loc['prc0.50','s2']
            #standard deviation
            self.sigma      = df_hparam.loc['prc0.50','sigma_vel']
    
        #read random term
        if not fname_dBr is None:
            #read random term
            df_dBr = pd.read_csv(fname_dBr)
            #median scaling
            self.prof_dBr    = df_dBr.loc[:,'param_dBr_med']
            self.prof_latlon = df_dBr.loc[:,['Lat','Lon']]

    
    # Scaling functions
    # ---   ---   ---   ---   ---
    def calcVs0(self, prof_vs30, prof_k, prof_n, z_star=2.5):
        '''
        Determine shear-wave velocity at zero depth.

        Parameters
        ----------
        prof_vs30 : real
            DESCRIPTION.
        prof_k : real
            DESCRIPTION.
        prof_n : real
            DESCRIPTION.
        z_star : real, optional
            DESCRIPTION. The default is 2.5.

        Returns
        -------
        prof_vs0 : TYPE
            DESCRIPTION.
        '''

        #convert profile parameters to numpy arrays
        prof_vs30 = np.array(prof_vs30).flatten()
        prof_k    = np.array(prof_k).flatten()
        prof_n    = np.array(prof_n).flatten()

        #compute a
        prof_a =-1/prof_n
        #compute Vs0
        prof_vs0 = np.array([(z_star + 1/k * np.log(1+k*(30-z_star))) if np.abs(n-1) < 1e-9 else (k*(a+1)*z_star + (1+k*(30-z_star))**(a+1) - 1) / ((a+1)*k) 
                             for v30, k, n, a in zip(prof_vs30, prof_k, prof_n, prof_a)])
        
        prof_vs0 *= prof_vs30/30
        return prof_vs0
    
    def calcK(self, vs30):
        
        return None

    def calcN(self, vs30):
        '''
        Determine curvature parameter.

        Parameters
        ----------
        vs30 : real
            Time-average shear-wave velocity at the top 30m.

        Returns
        -------
        n : real
            Curvature parameter.
        '''

        #vs30 scaling
        lnVs30s = (np.log(vs30)-self.logVs30mid)/self.logVs30scl
    
        #profile curvature parameter
        n   =  1 + self.s2 * self.sigmoid(lnVs30s)
        
        return n
    
    # Vs profile functions
    # ---   ---   ---   ---   ---
    def calcVs(self, vs0, k, n, sigma,  z_array, eps_array=None):
        '''
        Determine velocity profile.

        Parameters
        ----------
        vs0 : real or np.array
            Shear-wave velocity at zero depth.
        k : real or np.array
            Slope parameter.
        n : real or np.array
            Cruvature parameter.
        sigma : real or np.array
            aleatory standard deviation.
        z_array : np.array
            Depth array.
        eps_array : np.array, optional
            Epsilon array. The default is None.

        Returns
        -------
        vel_prof : np.array
            Velocity profile.
        '''
        
        #convert profile parameters to numpy arrays
        vs0 = np.array(vs0).flatten()
        k   = np.array(k).flatten()
        n   = np.array(n).flatten()
        
        #number of profiles
        n_p = len(vs0)
        
        #epsilon array
        if eps_array is None: eps_array = 0
        
        #initialize vel profile
        vel_prof = np.full((len(z_array),n_p), np.nan)
        #iterate over different profiles
        for j in range(n_p):
            vel_prof[:,j] = (vs0[j] * ( 1 + k[j] * ( np.maximum(0, z_array-self.z_star) ) )**(1/n[j])) * np.exp(eps_array*sigma) 
        
        return vel_prof
   
    # Auxiary functions
    # ---   ---   ---   ---   ---
    def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

# Stationary Model
# -----------------------------------------
class VelModelStationary(VelModel):
    def __init__(self, fname_hparam=None, z_star=2.5):
        
        #intialize scaling terms
        self.z_star = z_star
        
        #read model hyper-parameters
        if not fname_hparam is None:
            #read coefficient flatifle
            df_hparam = pd.read_csv(fname_hparam)
            #median scaling
            self.logVs30mid = df_hparam.loc['prc0.50','logVs30mid']
            self.logVs30scl = df_hparam.loc['prc0.50','logVs30scl']
            self.r1         = df_hparam.loc['prc0.50','r1']
            self.r2         = df_hparam.loc['prc0.50','r2']
            self.r3         = df_hparam.loc['prc0.50','r3']
            self.s2         = df_hparam.loc['prc0.50','s2']
            #standard deviation
            self.sigma      = df_hparam.loc['prc0.50','sigma_vel']
        else:
            #median scaling
            self.logVs30mid = 6.20947
            self.logVs30scl = 0.372021
            self.r1         =-2.32772
            self.r2         = 3.73403
            self.r3         = 0.2734575
            self.s2         = 4.83542
            #standard deviation
            self.sigma      = 0.376676

    # Scaling functions
    # ---   ---   ---   ---   ---
    def calcK(self, vs30):
        '''
        Determine slope parameter.

        Parameters
        ----------
        vs30 : real
            Time-average shear wave velocity at the top 30m.

        Returns
        -------
        k : real
            Slope parameter.
        '''
        
        #vs30 scaling
        lnVs30s = (np.log(vs30)-self.logVs30mid)/self.logVs30scl

        #profile slope parameter
        k   = np.exp( self.r1 + self.r2 * self.sigmoid(lnVs30s) + self.r3 * self.logVs30scl * np.log(1 + np.exp(lnVs30s)) )

        return k
    
    # User-interface functions
    # ---   ---   ---   ---   ---
    def Vs(self, vs30, z_array, eps_array=None):
        '''
        Return shear-wave velocity model.

        Parameters
        ----------
        vs30 : real
            Time-average shear wave velocity at the top 30m.
        z_array : np.array
            Velocity profile depth array.
        eps_array : np.array, optional
            epsilon array. The default is None.

        Returns
        -------
        vs_med : np.array
            Shear-wave velocity median profile.
        vs_sig : np.array
            Shear-wave velocity standard deviation.
        - - - - 
        vs_array : np.array
            Shear-wave veolicty model for a given epsilon
        '''
        
        #compute scaling parameters
        k   = self.calcK(vs30)
        n   = self.calcN(vs30)
        vs0 = self.calcVs0(vs30, k, n, self.z_star)
        
        #compute velocity profile
        if eps_array is None:
            vs_med = self.calcVs(vs0, k, n, self.sigma, z_array)
            vs_sig = np.full(vs_med.shape, self.sigma)
            
            return vs_med, vs_sig
        else:
            vs_array = self.calcVs(vs0, k, n, self.sigma, z_array, eps_array)
            
            return vs_array
        
# Spatially Varying Model
# -----------------------------------------
class VelModelSpatialVarying(VelModel):
    def __init__(self, fname_hparam=None, fname_dBr=None, z_star=2.5, utm_zone='11S'):
        
        #intialize scaling terms
        self.z_star = z_star
        
        #utm zone
        self.utm_zone = utm_zone
        #define projection system
        self.utm_proj = pyproj.Proj("+proj=utm +zone="+utm_zone[:2]+" +ellps=WGS84 +datum=WGS84 +units=km +no_defs")
        
        #read model hyper-parameters
        if not fname_hparam is None:
            #read coefficient flatifle
            df_hparam = pd.read_csv(fname_hparam)
            #median scaling
            self.logVs30mid = df_hparam.loc['prc0.50','logVs30mid']
            self.logVs30scl = df_hparam.loc['prc0.50','logVs30scl']
            self.r1         = df_hparam.loc['prc0.50','r1']
            self.r2         = df_hparam.loc['prc0.50','r2']
            self.r3         = df_hparam.loc['prc0.50','r3']
            self.s2         = df_hparam.loc['prc0.50','s2']
            #hyper-parameters for spatial variation
            self.ell_rdB    = df_hparam.loc['prc0.50','ell_rdB']
            self.omega_rdB  = df_hparam.loc['prc0.50','omega_rdB']
            #standard deviation
            self.sigma      = df_hparam.loc['prc0.50','sigma_vel']
        else:
            #median scaling
            self.logVs30mid = 6.20947
            self.logVs30scl = 0.372021
            self.r1         =-2.32772
            self.r2         = 3.73403
            self.r3         = 0.2734575
            self.s2         = 4.83542
            #hyper-parameters for spatial variation
            self.ell_rdB    = 1.46847
            self.omega_rdB  = 0.272678
            #standard deviation
            self.sigma      = 0.376676

        #read spatially varying parameters of training dataset
        if not fname_dBr is None:
            #read spatially varying terms
            df_dBr = pd.read_csv(fname_dBr)
            #coordinates of training dataset
            self.vprof_latlon  = df_dBr.loc[:,['Lat','Lon']].values
            #spatially varying terms
            self.vprof_dBr_med = df_dBr.loc[:,'param_dBr_med'].values
            self.vprof_dBr_sig = df_dBr.loc[:,'param_dBr_std'].values
        else:
            raise RuntimeError('Unspecified spatially varying parameters of training dataset.')

        #convert lat/lon to UTM coordinates
        self.vprof_XY = np.array( self.CalcCorrUTM(self.vprof_latlon[:,0], self.vprof_latlon[:,1]) ).T
                
    # Scaling functions
    # ---   ---   ---   ---   ---
    def calcK(self, vs30, latlon):
        '''
        Compute profile specific slope parameter

        Parameters
        ----------
        vs30 : real or np.array
            Time-average shear wave velocity at the top 30m..
        latlon : np.array
            Latitude longitude coordinates for velocity profiles.

        Returns
        -------
        k : real or np.array
            Profile specific slope parameter.
        k_glob : real or np.array
            Global slope parameter.
        dBr_mu : real or np.array
            Profile specific mean adjustment to slope parameter.
        dBr_sig : real or np.array
            Uncertainty of profile specific mean adjustment to slope parameter.
        dBr_cov : real or np.array
            Covariance of profile specific mean adjustment to slope parameter..
        '''
        
        #convert latlon to 2d array
        latlon = np.reshape(latlon, (-1, 2)) if latlon.ndim==1 else latlon
        
        #compute UTM coordinates
        XY = np.array( self.CalcCorrUTM(latlon[:,0], latlon[:,1]) ).T
        
        #vs30 scaling
        lnVs30s = (np.log(vs30)-self.logVs30mid)/self.logVs30scl

        #global profile slope parameter
        k_glob = np.exp( self.r1 + self.r2 * self.sigmoid(lnVs30s) + self.r3 * self.logVs30scl * np.log(1 + np.exp(lnVs30s)) )

        #sample dB_r term at new location
        dBr_mu, dBr_sig, dBr_cov = self.SampleCoeffs(XY, self.vprof_XY, self.vprof_dBr_med, self.vprof_dBr_sig,
                                                        self.ell_rdB, self.omega_rdB)

        #profile specific curvature coefficient
        k = k_glob * np.exp(dBr_mu)
        
        return k, k_glob, dBr_mu, dBr_sig, dBr_cov
    
    # User-interface functions
    # ---   ---   ---   ---   ---
    def Vs(self, vs30, z_array, latlon, eps_array=None, n_realiz=0):
        
        '''
        Return shear-wave velocity model.

        Parameters
        ----------
        vs30 : real
            Time-average shear wave velocity at the top 30m.
        z_array : np.array
            Velocity profile depth array.
        eps_array : np.array, optional
            epsilon array. The default is None.

        Returns
        -------
        vs_med : np.array
            Shear-wave velocity site-specific median profile.
        vs_glob : np.array
            Shear-wave velocity global median profile.
        vs_sig : np.array
            Shear-wave velocity standard deviation.
        - - - - 
        vs_array : np.array
            Shear-wave veolicty model for a given epsilon
        '''
        
        #convert lat long to numpy matrix
        latlon = np.array(latlon).flatten()
        
        #compute scaling parameters
        #slope
        k, k_glob, dBr_mu, dBr_sig, dBr_cov   = self.calcK(vs30, latlon)
        #random realizations
        if n_realiz > 0:
            #random samples
            dBr = self.MVNRnd(mean=dBr_mu, cov=dBr_cov, n_samp=n_realiz)
            #random slope parameters
            k = k_glob * np.exp(dBr)
        #curvature
        n = self.calcN(vs30)
        if n_realiz > 0:
            n = np.matlib.repmat(n, 1, n_realiz)

        #surface shear wave velocity (site specific)
        vs0      = self.calcVs0(vs30, k, n, self.z_star)      if n_realiz == 0 else self.calcVs0(npmat.repmat(vs30, 1, n_realiz), k, npmat.repmat(n, 1, n_realiz), self.z_star)
        #surface shear wave velocity (global)
        vs0_glob = self.calcVs0(vs30, k_glob, n, self.z_star) if n_realiz == 0 else self.calcVs0(npmat.repmat(vs30, 1, n_realiz), k_glob, npmat.repmat(n, 1, n_realiz), self.z_star)
        
        #compute velocity profile
        if eps_array is None:
            vs_med  = self.calcVs(vs0,      k,      n, self.sigma, z_array)
            vs_glob = self.calcVs(vs0_glob, k_glob, n, self.sigma, z_array)
            vs_sig  = np.full(vs_med.shape, self.sigma)
            
            return vs_med, vs_glob, vs_sig
        else:
            vs_array = self.calcVs(vs0, k, n, self.sigma, z_array, eps_array)
            
            return vs_array
 
    # Auxiary functions
    # ---   ---   ---   ---   ---
    def CalcCorrUTM(self, lat, lon):
        '''
        Compute UTM coordinates

        Parameters
        ----------
        lat : real or np.array
            Latitude coordinates.
        lon : real or np.array
            Longitude coordinates.

        Returns
        -------
        X : np.array
            Horizontal UTM coordinates.
        Y : np.array
            Vertical UTM coordinates.
        '''
        
        #convert latitude/longitude to numpy arrays
        lat = np.array(lat).flatten()
        lon = np.array(lon).flatten()
        
        #number of profiles
        n_p = len(lat)
        assert(len(lat)==len(lon)),'Error. Inconsistent number of latitude and longitude coordinates.'
        
        #compute utm coordinates
        XY = np.array([self.utm_proj(lon[j], lat[j])  for j in range(n_p)])  
        X  = XY[:,1] 
        Y  = XY[:,0]
        
        return X, Y

    def CalcCovNegExp(self, XY_1, XY_2,
                      hyp_ell= 0, hyp_omega= 0, hyp_pi = 0,
                      delta = 1e-9):
        "Compute negative exponential kernel matrix"
    
        #number of grid nodes
        n_pt_1 = XY_1.shape[0]
        n_pt_2 = XY_2.shape[0]
        
        #create cov. matrix
        cov_mat = np.zeros([n_pt_1,n_pt_2]) #initialize
        for j in range(n_pt_1):
            dist = scipylinalg.norm(XY_1[j] - XY_2[:,:],axis=1)
            cov_mat[j,:] = hyp_pi**2 + hyp_omega** 2 * np.exp(- dist/hyp_ell)
        
        if n_pt_1 == n_pt_2:
            for i in range(n_pt_1):
                cov_mat[i,i] += delta

        return cov_mat

    def SampleCoeffs(self, XY_new, XY_data, 
                     c_data_mu, c_data_sig = None,
                     hyp_ell = 0, hyp_omega = 0, hyp_pi = 0):
        '''
        Sample spatially varying terms at new locations conditioned on training ones

        Parameters
        ----------
        XY_new : np.array
            Coordinates of new locations.
        XY_data : np.array
            Coordinates of training location.
        c_data_mu : np.array
            Mean value of spatially varying coefficient in training dataset.
        c_data_sig : np.array, optional
            Standrad deviation of spatially varying coefficient in training dataset. The default is None.
        hyp_ell : real, optional
            Correlation length. The default is 0.
        hyp_omega : real, optional
            Scale. The default is 0.
        hyp_pi : real, optional
            Constant Scale. The default is 0.

        Returns
        -------
        c_new_mu : np.array
            Mean value of spatially varying coefficents at new locations.
        c_new_sig : np.array
            Standard deviation of spatially varying coefficents at new locations.
        c_new_cov : np.array
            Covaraince of spatially varying coefficents at new locations.

        '''

        #number of training data points
        n_pt_data = XY_data.shape[0]
        
        #mean value of training dataset
        c_data_mu = c_data_mu.flatten()
        
        #uncertainty of training dataset
        if c_data_sig is None: c_data_sig = np.zeros(n_pt_data)
        c_data_cov = np.diag(c_data_sig**2) if c_data_sig.ndim == 1 else c_data_sig
        assert( np.all(np.array(c_data_cov.shape) == n_pt_data) ),'Error. Inconsistent size of c_data_sig'
        
        #compute covariance between data 
        K      = self.CalcCovNegExp(XY_data, XY_data, hyp_ell, hyp_omega, hyp_pi, delta=1e-9)
        #covariance between data and new locations
        k      = self.CalcCovNegExp(XY_new,  XY_data, hyp_ell, hyp_omega, hyp_pi)
        #covariance between new locations
        k_star = self.CalcCovNegExp(XY_new,  XY_new,  hyp_ell, hyp_omega, hyp_pi)
        
        #inverse of covariance matrix
        K_inv = scipylinalg.inv(K)
        #product of k * K^-1
        kK_inv = k.dot(K_inv)
        
        #posterior mean and variance at new locations
        c_new_mu  = kK_inv.dot(c_data_mu)
        c_new_cov = k_star - kK_inv.dot(k.transpose()) + kK_inv.dot( c_data_cov.dot(kK_inv.transpose()) )
        #posterior standard dev. at new locations
        c_new_sig = np.sqrt(np.diag(c_new_cov))
        
        return c_new_mu, c_new_sig, c_new_cov

    def MVNRnd(self, mean=None, cov=None, seed=None, n_samp=None):
        '''
        Multivariate normal random generator

        Parameters
        ----------
        mean : np.array, optional
            Mean values. The default is None.
        cov : np.array, optional
            Covariance matrix. The default is None.
        seed : np.array, optional
            Seed standard normal distributed values. The default is None.
        n_samp : int, optional
            Number of random samples. The default is None.

        Returns
        -------
        samp : np.array
            Random samples.
        '''
                    
        #number of dimensions
        n_dim = len(mean) if not mean is None else cov.shape[0]
        assert(cov.shape == (n_dim,n_dim)),'Error. Inconsistent size of mean array and covariance matrix'
            
        #set mean array to zero if not given
        if mean is None: mean = np.zeros(n_dim)
    
        #compute L D L' decomposition
        L, D, _ = scipylinalg.ldl(cov)
        assert( not np.count_nonzero(D - np.diag(np.diagonal(D))) ),'Error. D not diagonal'
        assert( np.all(np.diag(D) > -1e-1) ),'Error. D diagonal is negative'
        #extract diagonal from D matrix, set to zero any negative entries due to bad conditioning
        d      = np.diagonal(D).copy()
        d[d<0] = 0
        #compute Q matrix
        Q = L @ np.diag(np.sqrt(d))
    
        #genereate seed numbers if not given 
        if seed is None: seed = np.random.standard_normal(size=(n_dim, n_samp))
    
        #generate random multi-normal random samples
        samp = Q @ (seed )
        samp += mean[:,np.newaxis] if samp.ndim > 1 else mean        
        
        return samp
