# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:52:24 2020

@author: lzampa
"""

# -----------------------------------------------------------------------------
import os
mdir = os.path.dirname( os.path.abspath(__file__) ) 

import imp
import numpy as np
utl = imp.load_source( 'module.name', mdir+os.sep+'utils.py' )
rt = imp.load_source( 'module.name', mdir+os.sep+'raster_tools.py' )
ga = imp.load_source( 'module.name', mdir+os.sep+'grav_anomaly_calc.py' )

s = os.sep
# -----------------------------------------------------------------------------
# Constants
G = utl.G
M = utl.M
a_wgs84 = utl.a_wgs84
c_wgs84 = utl.c_wgs84
R_wgs84 = utl.R_wgs84
J2_wgs84 = utl.J2_wgs84
w_wgs84 = utl.w_wgs84 

#------------------------------------------------------------------------------
# Module path
abs_path = os.path.abspath( __file__ )
mpath = os.path.dirname( abs_path )

# -----------------------------------------------------------------------------
def gobs2fa( gdb, dtm=None, herr=2.5, name='New_Gravity_DataBase', path=""):
    # -------------------------------------------------------------------------
    # Load database
    if type(gdb) == str:
        path = os.path.dirname(gdb)
        name = os.path.basename(gdb).split('.')[0]
        gdb = np.loadtxt(gdb, delimiter=',', skiprows=1)    
    
    if path != "": 
        os.makedirs( path, exist_ok=True ) 
    
    # -------------------------------------------------------------------------
    # Create Te_correction directory    
    Te_correction_dir = path +s+ 'Te_correction'
    os.makedirs( Te_correction_dir, exist_ok=True )  
    os.makedirs( Te_correction_dir +s+ 'dtm1', exist_ok=True ) 
    os.makedirs( Te_correction_dir +s+ 'dtm2', exist_ok=True ) 
    
    # -------------------------------------------------------------------------
    # Extract columns 
    lon = gdb[:,1]
    lat = gdb[:,0]
    gobs = gdb[:,2]
    hst = gdb[:,3]
    st_type = gdb[:,4]
    ref = gdb[:,5]
    
    nrows = lon.shape[0] # number of rows in the database
    
    if gdb.shape[1] == 7 :
        line = gdb[:,6]
    else:
        line = np.zeros(0)
    
    # -------------------------------------------------------------------------
    # Create station numbers
    st_num = np.arange(1, nrows+1)         
    
    # -------------------------------------------------------------------------
    # Extract dtm height from dtm and validate the mesured height based on herr
    hdiff = np.zeros(nrows)
    hnew = np.zeros(nrows)
    slab_c = np.zeros(nrows)
    curv_c = np.zeros(nrows)
    if dtm is None :
        hdtm = hst
        hnew = hst
    else :    
        hdtm,_,_ = rt.xy2rasterVal(dtm, lon, lat, close=True)
        hdiff = hst - hdtm
        idx = np.abs( hdiff ) >= herr 
        hnew[idx] = hdtm[idx]
        hnew[~idx] = hst[~idx]
        hnew[(st_type==1) & (hnew>=0)] = hst[(st_type==1) & (hnew>=0)]  
        hnew[(st_type==2) & (hnew>=0)] = hst[(st_type==2) & (hnew>=0)]  
        
    # -------------------------------------------------------------------------
    # Normal gravity 
    gn = ga.gn_84( lat )     

    # -------------------------------------------------------------------------
    # Atmpspheric correction
    a_c = ga.atm_c( hnew )   

    # -------------------------------------------------------------------------
    # Free Air / Water correction
    faw_c = np.zeros(nrows)
    faw_c[st_type==0] = ga.fa_c( hnew[st_type==0], lat[st_type==0], model='ell', R=R_wgs84 )
    faw_c[st_type==1] = 0      
    faw_c[st_type==2] = ga.fw_c( hnew[st_type==2], lat[st_type==2], model='sph' )
    
    # -------------------------------------------------------------------------
    # Bouguer slab ( BA )
    slab_c[ st_type==0] = ga.slab( hnew[ st_type==0 ], dc=2670 )  
    slab_c[ st_type==1 ] = ga.slab( hnew[ st_type==1 ], dc= -2670 + 1030 ) 
    slab_c[ st_type==2 ] = ga.slab( hnew[ st_type==2 ], dc= -2670 + 1030 ) 
    
    # -------------------------------------------------------------------------
    # Curvature correction ( BB )
    curv_c[st_type==0] = ga.curv_c( hnew[ st_type==0 ], st_type=0 )  
    curv_c[st_type==1 ] = ga.curv_c( hnew[ st_type==1 ], st_type=1 )     
    curv_c[st_type==2 ] = ga.curv_c( hnew[ st_type==2 ], st_type=2 ) 
    
    # -------------------------------------------------------------------------
    # Free Air / Water anomaly
    faw_a = gobs - ( gn + a_c + faw_c )
    
    # -------------------------------------------------------------------------
    # Simple Bouguer anomaly
    sbg_a = gobs - ( gn + a_c + faw_c + slab_c + curv_c )       
   
    # ------------------------------------------------------------------------
    # Save data 1
    gdb_new1 = np.column_stack( ( st_num, st_type, ref, lon, lat, hst, hdtm, gobs,
                                  gn, a_c, faw_c, slab_c, curv_c, faw_a, sbg_a, line ) )
    
    hcol = ['St_num', 'St_Type', 'Ref', 'Lon', 'Lat', 'H_St', 'H_Dtm', 'G_Obs', 
            'Gn', 'Atm_C', 'Faw_C', 'Slab_C', 'Curv_C', 'Faw_A', 'SBg_A', 'line']
    header1 = ''
    for i in hcol: header1 = header1+f'{i:>12},'
    header1 = header1[:-1]        
    fmt1 = '% 12d,% 12d,% 12d,% 12.6f,% 12.6f,% 12.2f,% 12.2f,% 12.3f,'+\
           '% 12.3f,% 12.3f,% 12.3f,% 12.3f,% 12.3f,% 12.3f,% 12.3f,% 12d'
    
    np.savetxt( path+s+name+'_Corr&An.csv', 
                gdb_new1, header=header1, fmt=fmt1, comments='')
    
    return gdb_new1

# -----------------------------------------------------------------------------
#if __name__ == '__main__':
#    
#    gdb = input('Set DataSet absolute path_name: \n')  
#    dtm1_list = input('Set directory with High Resolution DTMs')  
#    if dtm1_list == "": 
#        dtm1_list = mpath+s+'dtm1'
#
#             
#    gobs2fa( gdb, dtm1, herr=2.5, name='New_Gravity_DataBase', path="")
 