#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 09:08:21 2019

@author: glavrent
"""

#load libraries
import os
#arithmetic libraries
import numpy as np
#statistics
import pandas as pd
#user functions
def floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)

#-----------------------------------------------------------------------------
#USGS object to query the USGS velocity model v21
#-----------------------------------------------------------------------------
class USGSVelModelv21:
    
    def __init__(self,dir_velmodel,fname_velmodel,dir_geomodelgrids='/usr/local/geomodelgrids/geomodelgrids-1.0.0rc2-Linux_x86_64/'):
    #loads the USGS tools to query the velocity model
        #dir_velmodel: directory of velocity model
        #fname_velcmodel: file name of velocity model
        #fname_ext_velcmodel: file name of extended velocity model [optional]

        #input assertions
        assert isinstance(dir_velmodel,str), "Invalid type for dir_velmodel, it must be a string"
        assert isinstance(fname_velmodel,str), "Invalid type for fname_velmodel, it must be a string"
            

        # #calls/loads the USGS model in terminal
        # unix_cd = 'cd '+dir_velmodel
        # unix_call_USGStool = '. ./setup.sh'
        
        #store UNIX commands to call USGS vel model
        self.USGStool_int= f'cd {dir_geomodelgrids}\n. ./setup.sh'
        #store directory of velocity model
        self.fname_velmodel = dir_velmodel+fname_velmodel
        
        #define query files
        self.dirfn_input = '/tmp/usgs_vs_model_query_in.txt'
        self.dirfn_out   = '/tmp/usgs_vs_model_query_out.txt'
        self.dirfn_log   = '/tmp/usgs_vs_model_query_log.txt'
        
        #build query commands
        #velocity model
        self.vmodel_cmd = '--models='+self.fname_velmodel
        #input, output and log
        self.input_cmd  = f'--points={self.dirfn_input}'
        self.output_cmd = f'--output={self.dirfn_out}'
        self.log_cmd    = f'--log={self.dirfn_log}'
        

    #Processing methods
    #--------------------------------- 
    def QuerySurf(self, latlong):
    #query surface elevation
        #parse input
        lat = latlong[0]
        lon = latlong[1]
        
        #compile query 
        query_elev_cmd = f'geomodelgrids_queryelev {self.vmodel_cmd} {self.input_cmd} {self.output_cmd}'

        #find model elevation at lat long coordinates 
        #write input file with location info
        fid = open(self.dirfn_input,'w')
        fid.write(f'{lat} {lon}\n')
        fid.close()
        
        #query Vel model
        os.system(f'{self.USGStool_int} > /tmp/temp_out \n{query_elev_cmd}')
        
        #read output
        raw_data = pd.read_csv(self.dirfn_out, sep='[\s]{1,}',header=None, index_col=False, skiprows=2, engine='python')
        
        return float(raw_data.iloc[0,-1])

    def QueryElev(self, latlong, elev):
    #query USGS vel model at specific elevations
        #velocity model headers
        headers_loc   = ['lat','lon','elev']
        headers_param = ['Vp','Vs','den','Qp','Qs','fltbl_id','zn_id']
        
        #parse input
        lat = latlong[0]
        lon = latlong[1]
        
        #query values
        values_cmd  = '--values=Vp,Vs,density,Qp,Qs,fault_block_id,zone_id'
        #compile query 
        query_vel_cmd  = f'geomodelgrids_query {self.vmodel_cmd} {self.input_cmd} {self.output_cmd} {self.log_cmd} {values_cmd}'

        #convert elevation to an array
        elev = np.array([elev]).flatten()
        #floor at second decimal point
        elev = floor(elev,precision=2)

        #surface elevation
        elev_surf = self.QuerySurf(latlong)

        #location query
        latlon_q = np.tile(np.array([lat,lon]),[len(elev),1])
        #create input file
        np.savetxt(self.dirfn_input,np.concatenate((latlon_q,np.expand_dims(elev, axis=1)),axis = 1),fmt='%.5f %.5f %.4f')
        #query Vel model
        os.system(f'{self.USGStool_int} > /tmp/temp_out\n{query_vel_cmd}')
        #read output
        vel_profile = pd.read_csv(self.dirfn_out, sep='[\s]{1,}',header=None, index_col=False, skiprows=2, engine='python')
        vel_profile.columns = headers_loc + headers_param
        #depth form surface
        vel_profile.loc[:,'z2surf'] = elev_surf - vel_profile.loc[:,'elev']
        #add surface elevation
        vel_profile.loc[:,'elev_surf'] = elev_surf
        #exclude layers with Non available (-999) Vs
        idx_nan = vel_profile[vel_profile.Vs == -1e20].index
        vel_profile.loc[idx_nan,headers_param] = np.nan

        #return velocity and surface elevation
        return latlong, elev_surf, vel_profile
    
    def QueryZ(self, latlong, z):
    #query USGS vel model at specific depths
        #surface elevation
        elev_surf = self.QuerySurf(latlong)
        
        #convert depth to an array
        z = np.array([z]).flatten()
        
        #elevation query
        elev_q = elev_surf - z
        
        return self.QueryElev(latlong, elev_q)

    def QueryProf(self, latlong, dz=None, z_min=None):
    #query entire USGS vel profile
        #model resolution
        z_res  = np.array([-500.,-3000.,-3500.,-10000.])
        dz_res = np.array([25,50,100,200])    
    
       #surface elevation
        elev_surf = self.QuerySurf(latlong)

        #profile elev query
        if not dz is None:
            if z_min is None: z_min = np.min(z_res)
            elev_q = np.arange(np.floor(elev_surf/dz)*dz, z_min-0.1, -dz)
            elev_q = np.insert(elev_q, 0, elev_surf)
        else:
            i2keep = elev_surf >= z_res
            z_res, dz_res = z_res[i2keep], dz_res[i2keep]
            #add surface
            z_res = np.insert(z_res, 0, elev_surf)
            #determine elev query
            elev_q = np.hstack([np.arange(np.floor(z_res[j]/dz_res[j])*dz_res[j], z_res[j+1], -dz_res[j]) 
                            for j in range(len(dz_res))])
            #add surface and botton
            elev_q = np.insert(elev_q, 0, elev_surf) #add surface
            elev_q = np.append(elev_q,z_res[-1])     #append last point
      
        return self.QueryZ(latlong, elev_q)
    
    def Depth2Vel(self, vel_thres, latlon):
    #returns the depth to vel
    
        #query velocity model
        latlon, elev_s, vel_prof = self.Query(latlon)

        #id of layer crossing the threshold vs
        zid2vel = np.where(vel_prof.Vs >= vel_thres)[0][0]
        if (zid2vel == 0):
            return 0
        else:
            return vel_prof.z2surf[zid2vel - 1]
        
    def VelprofCalcVs30(self, vel_prof):
    #VelprofCalcVs30 calculates the Vs30 of the velocity profile defined in vel_prof
    #  Input Arguments:
    #       vel_prof (pandas data-frame): velocity profile
       
        #id of layer crossing the 30m depth
        z_id_30 = np.where(vel_prof.z2surf >= 30)[0][0]
    
        #velocity, depth and thickness arrays
        vs_array = vel_prof.Vs[:z_id_30+1]
        depth_array = vel_prof.z2surf[:z_id_30+1]
        depth_array[-1:] = 30 #set the bottom depth of the last layer to 30m 
        thick_array = np.append(depth_array[0] ,np.diff(depth_array)) #compute thickness of each layer

        #calculate time average Vs-30
        vs_30 = np.sum(thick_array) / np.sum(thick_array / vs_array)
        return vs_30

    #Output methods
    #--------------------------------- 
    def GetVs30(self, latlon):
    #returns the vs30 of the velocity profile
    
        #query velocity model
        latlon, elev_s, vel_prof = self.Query(latlon)
        
        return self.VelprofCalcVs30(vel_prof)
    
    def GetZ1_0(self, latlon):
    #returns the Z1.0 depth
        return self.Depth2Vel(1000, latlon)/1000 #convert to km

    def GetZ2_5(self, latlon):
    #returns the Z2.5 depth
        return self.Depth2Vel(2500, latlon)/1000 #convert to km






