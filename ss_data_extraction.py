# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:24:45 2020

@author: lzampa
"""
# -----------------------------------------------------------------------------
import os
mdir = os.path.dirname( os.path.abspath(__file__) ) 
input_path = mdir + os.sep + "ss"
cwd = os.getcwd()

import imp
import matplotlib.pyplot as plt
import numpy as np
import os

utl = imp.load_source( 'module.name', mdir+os.sep+'utils.py' )
shp = imp.load_source( 'module.name', mdir+os.sep+'shp_tools.py' )
lzplt = imp.load_source( 'module.name', mdir+os.sep+'plot.py' )
rt = imp.load_source( 'module.name', mdir+os.sep+'raster_tools.py' )

s=os.sep

# -----------------------------------------------------------------------------
def extract( limits, data_type='raster', grid_step=None, ply_coast=False, fill=False,
             buffer=None, plot=False, path=cwd, maxSearchDist=10, smoothingIterations=1, 
             out_prjcode=4326, filt_size=None, st_type=False ) :
    
    if ply_coast == True :
        ply_coast = mdir + s + 'ss' + s + 'GSHHS_c_L1_LZ.shp'  
        
    if path != mdir + s + 'ss' :
        os.makedirs( path, exist_ok=True )
    
    d = { 'grav' : mdir+s+'ss'+s+'grav_29.1.tif', 
          'err' : mdir+s+'ss'+s+'grav_error_28.1.tif'  }

    # -------------------------------------------------------------------------
    # Clip to Lim.
    if out_prjcode != None :
        for i, key in enumerate( d ) :   
            d[key] = rt.raster_warp( d[key], lim=limits, extension='tif',
                                     out_prjcode=out_prjcode, new_name=key+'_ss',
                                     new_path=path, method=rt.gdal.GRA_Average, 
                                     close=True )           
    # -------------------------------------------------------------------------
    # Resampling
    if grid_step != None :
        for i, key in enumerate( d ) : 
            res0 = rt.raster_res( d['grav'], mean=True ) 
            if grid_step >= res0 :
                warp_method = rt.gdal.GRA_Average
            else :
                warp_method = rt.gdal.GRA_Bilinear
                
            d[key] = rt.raster_warp( d[key], lim=limits, xRes=grid_step, 
                                     yRes=grid_step, method=warp_method, extension='tif',
                                     new_name=rt.raster_name( d[key] )+'_res',
                                     new_path=rt.raster_path( d[key] ), close=True )
            
    # -------------------------------------------------------------------------
    # Resampling filt
    if filt_size != None :
        for i, key in enumerate( d ) : 
                
            d[key] = rt.resample_filt( d[key], resampling_factor=filt_size, close=True,
                                       path=path, suffix='_filt', overwrite=False,
                                       new_name=rt.raster_name( d[key] ) )
            
    # -------------------------------------------------------------------------
    # Mask land areas    
    if type( ply_coast ) == str :
        ply_coast = shp.translate( ply_coast, lim=limits, buffer=buffer, new_path=path, 
                                   suffix='_clip' )
        for i, key in enumerate( d ) : 
            d[key] = rt.rasterize( d[key], ply_coast, close=True, 
                                   path=rt.raster_path( d[key] ) )              
            
    # -------------------------------------------------------------------------
    # Fill Nans
    if fill != False :
        for i, key in enumerate( d ) : 
                
            d[key] = rt.raster_fill( d[key], maxSearchDist=maxSearchDist, smoothingIterations=smoothingIterations, 
             copy=True, path=path, new_name=rt.raster_name( d[key]), suffix='_fill', extension='tif' )              
    
    # -------------------------------------------------------------------------
    d['x'], d['y'], _ = rt.raster2array( d['grav'], midpx=True, close=True )                                      
    # -------------------------------------------------------------------------    
    
    # -------------------------------------------------------------------------
    # Add Point type : Sea_type / Land_type   
    if st_type == True :
        if type( ply_coast ) != str :
            ply_coast = mdir + s + 'ss' + s + 'GSHHS_c_L1_LZ.shp'          
        ply_coast = shp.translate( ply_coast, lim=limits, buffer=buffer, new_path=path, 
                                   suffix='_clip' )
        d['st_type'] = rt.raster_mask( d['grav'], shp_ply=ply_coast, plot=False )        
    
    # -------------------------------------------------------------------------  
    # PLot
    if plot is True :
        rt.pltr( d['grav'] )
    
    # -------------------------------------------------------------------------
    # Return
    if data_type == 'grid' :
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :
               d[key] = rt.raster2array( d[key], nodata=np.nan )[2]
               
    if data_type == 'vector' :
        d['x'] =  rt.raster2array( d['grav'], out_fmt='points', nodata=np.nan)[0]  
        d['y'] =  rt.raster2array( d['grav'], out_fmt='points', nodata=np.nan)[1]         
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :
               d[key] = rt.raster2array( d[key], out_fmt='points', nodata=np.nan)[2]  
        
    return d



