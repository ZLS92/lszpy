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
shp = imp.load_source( 'module.name', mdir+os.sep+'shp_tools.py' )
ga = imp.load_source( 'module.name', mdir+os.sep+'gravmag_processing.py' )

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
def gobs2fa( gdb, dtm=None, herr=2.5, name='New_Gravity_DataBase', path="", shp_write=True,
             x_c=1, y_c=0, g_c=2, h_c=3, st_type_c=None, ref_c=None, line_c=None,
             suffix='_Corr&An' ):
    
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
    lon = gdb[:,x_c]
    lat = gdb[:,y_c]
    gobs = gdb[:,g_c]
    hst = gdb[:,h_c]
    
    if st_type_c != None :
        st_type = gdb[:,st_type_c]
    else :
        st_type = np.zeros( hst.shape )
        
    if ref_c != None :
        ref = gdb[:,ref_c]  
    else :
        st_type = np.ones( lon.shape )        
    
    if line_c != None :
        line = gdb[:,line_c]
    else:
        print( 'no line column')
        line = np.zeros( hst.shape )
        
    nrows = hst.shape[0]    
    
    # -------------------------------------------------------------------------
    # Create station numbers
    st_num = np.arange(1, nrows+1)         
    
    # -------------------------------------------------------------------------
    # Extract dtm height from dtm and validate the mesured height based on herr
    hdiff = np.zeros(nrows)
    hnew = np.zeros(nrows)
    if dtm is None :
        hdtm = hst
        hnew = hst
        hdiff = hst - hst 
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
    a_c = np.zeros( nrows )
    a_c[st_type==0] = ga.atm_c( hnew[st_type==0] )
    a_c[st_type==1] = ga.atm_c( 0 )      
    a_c[st_type==2] = ga.atm_c( 0 )    

    # -------------------------------------------------------------------------
    # Free Air / Water correction
    faw_c = np.zeros(nrows)
    faw_c[st_type==0] = ga.fa_c( hnew[st_type==0], lat[st_type==0], model='ell', R=R_wgs84 )
    faw_c[st_type==1] = 0      
    faw_c[st_type==2] = ga.fw_c( hnew[st_type==2], lat[st_type==2], model='sph' )
    
    # -------------------------------------------------------------------------
    # Free Air / Water anomaly
    faw_a = gobs - ( gn + a_c + faw_c )  
   
    # ------------------------------------------------------------------------
    # Save data 
    gdb_new = np.column_stack( ( st_num, st_type, ref, line, lon, lat, hst, hdtm, hdiff, 
                                 hnew, gobs, gn, a_c, faw_c, faw_a ) )
    
    hcol = ['St_num', 'St_Type', 'Ref', 'Line', 'Lon', 'Lat', 'H_St', 'H_Dtm', 'H_Diff', 
            'H_New', 'G_Obs', 'Gn', 'Atm_C', 'Faw_C', 'Faw_A' ]

    fmt = [ '% 12d', '% 12d', '% 12d', '% 12d', '% 12.6f', '% 12.6f', '% 12.2f', '% 12.2f', '% 12.2f',
            '% 12.2f', '% 12.3f', '% 12.3f', '% 12.3f', '% 12.3f', '% 12.3f' ]  
    
    out_file = path +s+ name + suffix + '.csv'
    
    utl.array2csv( gdb_new, headers=hcol, sep=',', fmt=fmt, path_name=out_file )
                
#    with open( out_file, 'r' ) as f : 
#        print( f.read() )
     
    # Create dictionary -------------------------------------------------------    
    f = open( out_file, "r", encoding="utf8" )
    lines = f.readlines()
    data = []
    for i, l in enumerate( lines ) :
        lnw = l.translate( str.maketrans( '', '', ' \n\t\r' ) ) # delete all withe spaces from string
        if i ==0 :
            keys = lnw.split(',')
        if i !=0 :
            data.append( list( np.float_( lnw.split(',') ) ) )
    data = np.asarray( data )
    gdict = {}
    for i, k in enumerate( keys ) :
        gdict[ k ] = data[:,i] 
    f.close()    
        
    # Create shapefile -------------------------------------------------------- 
    if shp_write is True :
        shp.write_points( lon, lat, fields=gdict, prj_code=4326, 
                          name=name+suffix+'_shp', path=path )
    
    # Return ------------------------------------------------------------------
    return gdict, out_file, gdb_new, hcol, fmt

 