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

#-----------------------------------------------------------------------------
#USGS object to query the USGS velocity model
#-----------------------------------------------------------------------------
class USGSVelModel:
    
    def __init__(self,dir_velmodel,fname_velmodel,fname_ext_velmodel=None):
    #loads the USGS tools to query the velocity model
        #dir_velmodel: directory of velocity model
        #fname_velcmodel: file name of velocity model
        #fname_ext_velcmodel: file name of extended velocity model [optional]

        #input assertions
        assert isinstance(dir_velmodel,str), "Invalid type for dir_velmodel, it must be a string"
        assert isinstance(fname_velmodel,str), "Invalid type for fname_velmodel, it must be a string"
            

        #calls/loads the USGS model in terminal
        unix_cd = 'cd '+dir_velmodel
        unix_call_USGStool = '. ./setup.sh'
        
        #store UNIX commands to call USGS vel model
        self.unix_cmd = f'{unix_cd}\n{unix_call_USGStool}'
        #store directory of velocity model
        self.fname_velmodel = dir_velmodel+fname_velmodel
        #store directory of extended velocity model is specified
        if not fname_ext_velmodel is None:
            assert isinstance(fname_ext_velmodel,str), "Invalid type for fname_ext_velmodel, it must be a string"
            self.flag_ext_vel = True
            self.fname_ext_velmodel = dir_velmodel+fname_ext_velmodel
        else:
            self.flag_ext_vel = False
        
    #Processing methods
    #--------------------------------- 
    def Query(self,latlong):
    #query USGS model

        #define query files
        dirfn_input = '/tmp/usgs_vs_model_query_in.txt'
        dirfn_out = '/tmp/usgs_vs_model_query_out.txt'
        dirfn_log = '/tmp/usgs_vs_model_query_log.txt'
        #model resolution
        z_res = [-400,-3200,-6400,-45000]
        #z_res = [-400,-3200,-6400,-10000]
        dz_res = [25,50,100,200]
        dz_top = 12.5
        #velocity model headers
        headers = ['lon','lat','elev','Vp','Vs','den','Qp','Qs','z2surf','Fltbl_id','Zn_id','elev_surf']
        
        #parse input
        lat = latlong[0]
        lon = latlong[1]

        #build query commands
        #input, output and log
        input_cmd = f'-i {dirfn_input}'
        output_cmd = f'-o {dirfn_out}'
        log_cmd = f'-l {dirfn_log}'
        #database
        data_cmd = '-d '+self.fname_velmodel
        #database extended model
        if self.flag_ext_vel:
            data_ext_cmd = '-e '+self.fname_ext_velmodel
        else:
            data_ext_cmd = ''
        #query type
        query_type_cmd = '-t maxres'
        mem_cmd = '-c 1000'
        #compile query 
        query_cmd = f'cencalvmquery {input_cmd} {output_cmd} {log_cmd} {data_cmd} {query_type_cmd} {data_ext_cmd} {mem_cmd}'
        
        #find model elevation at lat long coordinates 
        #write input file with one point at the fine resolution region
        fid = open(dirfn_input,'w')
        fid.write(f'{lon} {lat} -350\n')
        fid.close()
        #query Vel model
        os.system(f'{self.unix_cmd} > /tmp/temp_out\n{query_cmd}')
        #os.system(f'{unix_cmd} > /tmp/temp_out\n{query_cmd}')
        #read output
        raw_data = pd.read_csv(dirfn_out, sep='[\s]{1,}',header=None, index_col=False, engine='python')
        elev_surf = float(raw_data.iloc[0,-1])
        
        #query velocity model
        #create input file
        z_q = np.arange(np.floor((elev_surf-dz_top)/dz_res[0])*dz_res[0],z_res[0],-dz_res[0])
        for i in range(1,len(z_res)):
            z_q = np.append(z_q, np.arange(z_res[i-1],z_res[i],-dz_res[i]))
        z_q = np.append(z_q,z_res[-1]) #append last point
        lonlat_q = np.tile(np.array([lon,lat]),[len(z_q),1])
        #create input file
        np.savetxt(dirfn_input,np.concatenate((lonlat_q,np.expand_dims(z_q, axis=1)),axis = 1),fmt='%.5f %.5f %.2f')
        #query Vel model
        os.system(f'{self.unix_cmd} > /tmp/temp_out\n{query_cmd}')
        #read output
        vel_profile = pd.read_csv(dirfn_out, sep='[\s]{1,}',header=None, index_col=False, engine='python')
        vel_profile.columns = headers
        #exclude layers with Non available (-999) Vs
        ids2drop = vel_profile[vel_profile.Vs == -999].index
        vel_profile = vel_profile.drop(ids2drop)
        vel_profile = vel_profile.reset_index()
        
        #return velocity and surface elevation
        return latlong, elev_surf, vel_profile
    
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






