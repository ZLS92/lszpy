# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:24:45 2020

Extract dtu-altimetry data from netcdf files
Data download from web_page : 
https://ftp.space.dtu.dk/pub/

@author: lzampa
"""
import os
s = os.sep
mdir = os.path.dirname( os.path.abspath(__file__) ) 
input_path = mdir + s + "dtu"
cwd = os.getcwd()

import imp
import numpy as np
from scipy import io
#import netCDF4 as nc

utl = imp.load_source( 'module.name', mdir+os.sep+'utils.py' )
shp = imp.load_source( 'module.name', mdir+os.sep+'shp_tools.py' )
rt = imp.load_source( 'module.name', mdir+os.sep+'raster_tools.py' )
lzplt = imp.load_source( 'module.name', mdir+os.sep+'plot.py' )

# -----------------------------------------------------------------------------
def dtu_netcdf2xyz( parameter, lim ) :  
    
    if parameter == 'grav' :
        file = input_path + s + 'DTU13GRA_1min.gra.nc'
    if parameter == 'err' :
        file = input_path + s + 'DTU13ERR_1min.err.nc'
    if parameter == 'mss' :
        file = input_path + s + 'DTU13MSS_1min.mss.nc'
    if parameter == 'mdt' :
        file = input_path + s + 'DTU13MDT_1min.mdt.nc'        
    
    p = io.netcdf.netcdf_file( file, mode='r', maskandscale=True )
    
    pp = p.variables[parameter]
    lats = p.variables['lat'][:] 
    lons = p.variables['lon'][:]     
    
    lons = lons[1:-1]
    pp = pp[:, 1:-1]
    lons = np.concatenate( ( lons[ int( (lons.shape[0])/2 ): ]-360, lons[ 0:int( (lons.shape[0])/2 ) ] ) )
    pp = np.column_stack( ( pp[ :, int( (pp.shape[1])/2 ): ], pp[ :, 0:int( (pp.shape[1])/2 ) ] ) )
    
    ymin = np.where( ( lats>=lim[2] ) & ( lats<=lim[3] ) )[0].min()
    ymax = np.where( ( lats>=lim[2] ) & ( lats<=lim[3] ) )[0].max()
    xmin = np.where( ( lons>=lim[0] ) & ( lons<=lim[1] ) )[0].min()
    xmax = np.where( ( lons>=lim[0] ) & ( lons<=lim[1] ) )[0].max()
    y = lats[ ymin : ymax+1 ]
    x = lons[ xmin : xmax+1 ]
    
    xx, yy = np.meshgrid( x , y ) 
    
    pp = pp[ ymin : ymax+1, xmin : xmax+1 ] 
    
    yy = np.flipud( yy )
    pp = np.flipud( pp )
    
    return  xx, yy, pp

# -----------------------------------------------------------------------------
def extract( limits, data_type='raster', grid_step=None, ply_coast=False,
             filt=None, fill=False, maxSearchDist=10, smoothingIterations=1,
             buffer=None, plot=False, nodata=9999, path=mdir, filt_size=None ) :
    
    if ply_coast == True :
        
        ply_coast = mdir + os.sep + 'dtu' + os.sep + 'GSHHS_c_L1_LZ.shp'  
        
    if path != mdir :
        os.makedirs( path, exist_ok=True )
    
    d = { 'grav' : None, 'err' : None, 'mss' : None, 'mdt' : None }
       
    # -------------------------------------------------------------------------
    # Extract    
    for i, key in enumerate( d ) :
        if i == 0 :
            x, y, d[key] = dtu_netcdf2xyz( key, lim=limits )
        else :
            d[key] = dtu_netcdf2xyz( key, lim=limits )[2]    
    d['x'], d['y'] = x, y   
           
    # -------------------------------------------------------------------------
    # Grid to Raster
    for i, key in enumerate( d ) : 
        if key not in ( 'x', 'y' ) :
            d[key] = rt.xyz2raster( d['x'], d['y'], d[key], new_name='dtu_'+key, 
                                    path=path, extension='tif', prjcode=4326, 
                                    nodata=nodata, close=True )

    # -------------------------------------------------------------------------
    # Resampling
    if grid_step != None :
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :
                res0 = rt.raster_res( d['grav'], mean=True ) 
                if grid_step >= res0 :
                    warp_method = rt.gdal.GRA_Average
                else :
                    warp_method = rt.gdal.GRA_Bilinear
                    
                d[key] = rt.raster_warp( d[key], lim=limits, xRes=grid_step, 
                                         yRes=grid_step, method=warp_method, extension='tif',
                                         new_name=rt.raster_name( d[key] ) +'_res',
                                         new_path=rt.raster_path( d[key] ), close=True )
                
        d['x'], d['y'], _ = rt.raster2array( d['grav'], midpx=True, close=True ) 

    # -------------------------------------------------------------------------
    # Mask land areas    
    if type( ply_coast ) == str :
        ply_coast = shp.translate( ply_coast, lim=limits, buffer=buffer, new_path=path, suffix='_clip' )
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :            
                d[key] = rt.rasterize( d[key], ply_coast, close=True, 
                                       path=rt.raster_path( d[key] ) ) 

    # -------------------------------------------------------------------------
    # Resampling filt
    if filt_size != None :
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :     
                d[key] = rt.resample_filt( d[key], resampling_factor=filt_size, close=True,
                                           path=path, suffix='_filt', overwrite=False,
                                           new_name=rt.raster_name( d[key]) )    

    # -------------------------------------------------------------------------
    # Fill Nans
    if fill != False :
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :     
                d[key] = rt.raster_fill( d[key], maxSearchDist=maxSearchDist, smoothingIterations=smoothingIterations, 
                 copy=True, path=path, new_name=rt.raster_name( d[key]), suffix='_fill', extension='tif' )                 

    # -------------------------------------------------------------------------
    # PLot
    if plot == True :
        rt.pltr( d['grav'] )
    
    # -------------------------------------------------------------------------
    # Return
    d['x'] = rt.raster2array( d['grav'], out_fmt='array' )[0]
    d['y'] = rt.raster2array( d['grav'], out_fmt='array' )[1]

    if data_type == 'grid' :
        for i, key in enumerate( d ) : 
            if key not in ( 'x', 'y' ) :
               d[key] = rt.raster2array( d[key], nodata=np.nan )[2]
               
    if data_type == 'vector' :
        d['x'] =  rt.raster2array( d['grav'], out_fmt='points', nodata=np.nan )[0]  
        d['y'] =  rt.raster2array( d['grav'], out_fmt='points', nodata=np.nan )[1]          
        for i, key in enumerate( d ) :
            if key not in ( 'x', 'y' ) :
               print( rt.raster_nodata( d[key] ) )
               d[key] = rt.raster2array( d[key], out_fmt='points', nodata=np.nan )[2]

    return d
        