# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:26:35 2019

@author: lzampa
"""

# -----------------------------------------------------------------------------
import os
mdir = os.path.dirname( os.path.abspath(__file__) ) 
s = os.sep

# import imp
import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
from copy import copy
import time
import lszpy.utils as utl
import lszpy.shp_tools as shp
# utl = imp.load_source( 'module.name', mdir+os.sep+'utils.py' )
# shp = imp.load_source( 'module.name', mdir+os.sep+'shp_tools.py' )
t=time.time

# -----------------------------------------------------------------------------
# Constants
G = utl.G
M = utl.M
a_wgs84 = utl.a_wgs84
c_wgs84 = utl.c_wgs84
R_wgs84 = utl.R_wgs84
J2_wgs84 = utl.J2_wgs84
w_wgs84 = utl.w_wgs84

# -----------------------------------------------------------------------------
raster_globe = mdir +s+ 'gis' +s+ 'raster' +s+ 'raster_globe.tiff'
bat_eu_emodnet = mdir +s+ 'gis' +s+ 'raster' +s+ 'EU_SEA_EMODnet_2018_w84.tif'

# -----------------------------------------------------------------------------
def tmi(t1, t2=None):
    """
    Print time interval from time t1 to t2
    --- If "t2 = None" , then, "t2 = now"
    """
    if t2 ==None:
        print(t() - t1 )
    else:
        print( t2 - t1 )

# -----------------------------------------------------------------------------
def raster_filt( raster, radius=3, sigma=1, padw=10, pmode='linear_ramp',
                 plot=False, vmin=None, vmax=None, close=False, iter=1, 
                 ftype='mean', factor=2, overwrite=False, 
                 path_name='/vsimem/raster_filt' ):

    if type(raster) == str:
        raster = gdal.Open( raster )
        close = True
        
    if overwrite == False :
        ds = gdal.GetDriverByName( 'GTiff' ).CreateCopy( '/vsimem/raster_filt', raster, 0 )
    else :    
        ds = raster

    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    array[array==nodata]=np.nan
    ds_array = utl.filt2d( array, radius=radius, padw=padw, pmode=pmode, 
                           plot=plot, vmin=None, vmax=None, iter=iter, ftype=ftype, 
                           sigma=sigma, factor=factor ) 
              
    ds_array[array==nodata]=nodata
    band.WriteArray(ds_array)

    if close == True:
        path_name = ds.GetDescription()
        ds = None 
        raster = None
        ds = path_name

    return ds

# -----------------------------------------------------------------------------
def raster_copy( src_ds, new_name='copy', extension='tif', suffix='',
                 path=None, driver='GTiff', dtype=gdal.GDT_Float32,
                 close=False ) :

    if type(src_ds) == str:
        src_ds = gdal.Open(src_ds)

    driver = gdal.GetDriverByName('GTiff')
    if type(src_ds)==str:
        src_ds = gdal.Open(src_ds)

    if path not in ( None, '/vsimem/' ) :
        os.makedirs( path, exist_ok=True)
        new_path_name = path +os.sep+ new_name+ suffix+ '.'+ extension
    else:
        new_path_name = '/vsimem/'+new_name+'.'+extension

    gt = src_ds.GetGeoTransform()
    srs = src_ds.GetProjection()
    col = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    bands = src_ds.RasterCount
    ndv = src_ds.GetRasterBand(1).GetNoDataValue()

    src_ds_copy = driver.Create(new_path_name, col, rows, bands, dtype)
    src_ds_copy.SetGeoTransform( gt )
    src_ds_copy.SetProjection( srs )

    for band in range( 1, bands + 1 ):
        array = src_ds.GetRasterBand( band ).ReadAsArray()
        src_ds_copy.GetRasterBand( band ).WriteArray( array )
        src_ds_copy.GetRasterBand( band ).SetNoDataValue(ndv)
        
    if close == True:
        path_name = src_ds_copy.GetDescription()
        gdal.Unlink( src_ds.GetDescription() )
        gdal.Unlink( src_ds_copy.GetDescription() )
        src_ds_copy = None 
        src_ds = None
        src_ds_copy = path_name

    return src_ds_copy

# -----------------------------------------------------------------------------
def xy2rasterIdx( raster, x, y, prjcode=4326, close=False ):
    """
    Get pixel index at specific map coordinates (x, y)
    """
    if type(raster) == str :
        raster = gdal.Open(raster)
        close = True
        
    rprj = raster_prjcode( raster )
    
    if rprj != utl.prj_( prjcode ).srs :
        x, y = utl.prjxy( prjcode, rprj, x, y )

    if type( x ) in ( int, float ) :
        x, y = [x], [y]
        
    gt = raster.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    col = []
    row = []

    for i, j in zip( x, y ):
        px, py = map(int, gdal.ApplyGeoTransform( inv_gt, i, j ) )
        col.append(px)
        row.append(py)

    if close == True: 
        gdal.Unlink( raster.GetDescription() )
        raster = None

    return row, col

# -----------------------------------------------------------------------------
def xy2rasterVal( raster, x, y, prjcode=4326, band=1, close=False ) :
    """
    Extract gdal_raster values at x, y coordinates
    Return (1) Raster value (2) 
    """
    if type( raster ) == str:
        raster = gdal.Open( raster )
        close = True

    r_array = raster.GetRasterBand( band ).ReadAsArray()
    
    if type(x) not in ( int, float ) :
        x = x.ravel()
    if type(y) not in ( int, float ) :
        y = y.ravel()
        
    row, col = xy2rasterIdx( raster, x, y, prjcode=prjcode )
#    row, col = xy2rasterIdx( raster, x, y, prjcode=prjcode )

    r_val = r_array[ row, col ]

    if close == True: 
        path_name = raster.GetDescription()
        raster = None
        raster = path_name

    if type(x) not in ( int, float ) :    
        r_val = r_val.reshape( x.shape )
    
    return r_val, row, col

# -----------------------------------------------------------------------------
def raster_AddGcp( raster, x, y, z=0, prjcode=4326, close=False ):
    
    if type(raster) == str:
        raster = gdal.Open( raster )
        close = True    

    rprj = raster_prjcode( raster )
    
    if rprj != utl.prj_( prjcode ).srs :
        x, y = utl.prjxy( prjcode, rprj, x, y )
    
    r_val, row, col = xy2rasterVal( raster, x, y, prjcode=rprj )
    
    if type( z ) in ( int, float ) :
        z = np.full( np.size( x ), z )
    if ( type( z ) == str ) and ( z == 'raster' ) :
        z = r_val
    
    gcp_list=[]
    
    for i in range( np.size(x) ):
        gcp_list.append(gdal.GCP(x[i], y[i], z[i], float(col[i]), float(row[i])))
        
    wkt = raster.GetProjection()
    raster.SetGCPs(gcp_list, wkt)    

    if close == True: 
        path_name = raster.GetDescription()
        raster = None
        raster = path_name

    return raster

# -----------------------------------------------------------------------------
def raster_count( raster, close=False ) :

    if type( raster ) == str:
        raster = gdal.Open( raster )
        close = True
        
    count = raster.RasterCount
    
    if close==True:
        # path_name = raster.GetDescription()
        raster = None
    
    return count
    
# -----------------------------------------------------------------------------
def raster_edit( raster, x, y, z, prjcode=4326, band=1, close=False,
                 new_name=None, extension='tif', suffix='', path=None ) :                   
    """
    Change raster values at specific locations  
    Return : (1) modified raster, (2) differences with previus values 
    """
    
    if type( raster ) == str:
        raster = gdal.Open( raster )
        close = True

    r_array = raster.GetRasterBand( band ).ReadAsArray()

    r_val, row, col = xy2rasterVal( raster, x, y, prjcode=prjcode )

    r_array[ row, col ] = z
    
    if new_name is None :
        new_name = raster_name( raster )
            
    if path is None :
        path = raster_path( raster )
        
    new_raster = raster_write( raster, r_array, band=band, copy=True, new_name=new_name,
                               extension=extension, suffix=suffix, path=path )
    
    if close==True:
        # path_name = raster.GetDescription()
        path_name_new = new_raster.GetDescription()
        new_raster = None
        raster = None
        new_raster = path_name_new

    return new_raster

# -----------------------------------------------------------------------------
def raster2xyz( raster, nodata=False, condition=None, sep=False, midpx=False,
                plot=False, cmap='rainbow', vmin=None, vmax=None, s=None, close=False):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    xx,yy,zz = raster2array(raster, midpx=midpx)
    band = raster.GetRasterBand(1)
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
    xyz = np.c_[x,y,z]
    nan = band.GetNoDataValue()

    if nodata==False:
        xyz, z = xyz[z!=nan], z[z!=nan]
    if condition!=None:
        xyz=xyz[condition(z)]
    if sep==True:
        xyz=(xyz[:,0],xyz[:,1],xyz[:,2])

    Min, Max, Mean, Std = np.nanmin(xyz[:,2]),np.nanmax(xyz[:,2]),\
                          np.nanmean(xyz[:,2]),np.nanstd(xyz[:,2])

    print (f'Min:{Min} Max:{Max} Mean:{Mean} Std:{Std}')

    if plot==True:
        if vmin==None:
            vmin=Mean-2*Std
        if vmax==None:
            vmax=Mean+2*Std
        plt.scatter( xyz[:,0],xyz[:,1], c=xyz[:,2], s=s, cmap=cmap,
                     vmin=vmin, vmax=vmax)
        plt.colorbar()

    if close==True:
        raster=None

    return xyz

# -----------------------------------------------------------------------------
def raster2array( raster, midpx=False, nodata=None, out_fmt='array', prj_out=None,
                  close=False, band=1, array_type='float64'):
    """
    Extract xx yy zz arrays from gdal_raster grid.
    Input :
        raster --- Gdal raster object or file_name_stirng of input raster
    Point position :
    1 - If "midpx = False", extracts coordinates of cells up-left corner
    2 - If "midpx = True", extracts coordinates of cells centre
    Output type :
    1 - If "out_fmt = array", the output consist of 3 2D numpy arrays (xx, yy, zz)
        If "nodata = np.nan", nodada values are changed in numpy nans in the zz array
    2 - If "out_fmt = points", the output consist of 3 1D numpy arrays (x, y, z)
        If "nodata = np.nan", nodada values are removed from the 1D arrays
    Output coordinate projection :
       prj_out --- defines the crs of the output points (x, y).
                   It can be an integer EPSG (i.e. 4326) or a proj4 string.
    """

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    gt = raster.GetGeoTransform()
    zz = raster.GetRasterBand(band).ReadAsArray().astype('float64')
    nondata = raster.GetRasterBand(band).GetNoDataValue()

    if nodata is not None:
        zz[zz==nondata] = nodata

    dx = gt[1]
    dy = gt[5]
    nx = raster.RasterXSize
    ny = raster.RasterYSize

    if midpx == True:
        sx, sy, Nx, Ny = dx/2, dy/2, nx-1, ny-1
    if midpx == False:
        sx, sy, Nx, Ny = 0, 0, nx, ny

    xstart, xstop = gt[0]+sx, gt[0]+sx+Nx*dx
    ystart, ystop = gt[3]-sy, gt[3]-sy+Ny*dy
    x = np.linspace(xstart, xstop, nx)
    y = np.linspace(ystart, ystop, ny)
    xx, yy = np.meshgrid( x, y, copy=False )

    if out_fmt == 'points':
        xx = xx[~np.isnan(zz)].ravel()
        yy = yy[~np.isnan(zz)].ravel()
        zz = zz[~np.isnan(zz)].ravel()
#    if out_fmt == 'array':
#        yy = np.flipud( yy )

    if prj_out is not None:
        prj_in = raster_epsg( raster )
        xx, yy = utl.prjxy( prj_in, prj_out, xx, yy )

    if close==True: 
        raster = None

    return xx, yy, zz

# -----------------------------------------------------------------------------
def raster_lim(raster, out='xlyl', prjcode_out=None, midpx=False, close=False, m2km=False):
    """
    Extract raster limits:
    (lon_min, lon_max, lat_min, lat_max)
    """

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    prjcode_raster = raster_epsg(raster)
    gt = raster.GetGeoTransform()

    dx, dy = gt[1], gt[5]
    if midpx == True:
        sx, sy = dx/2, dy/2
    if midpx == False:
        sx, sy = 0, 0
    nx = raster.RasterXSize
    ny = raster.RasterYSize
    x0_in, x1_in = gt[0]+sx, gt[0]+sx+nx*dx
    y1_in, y0_in = gt[3]-sy, gt[3]-sy+ny*dy

    if prjcode_out == None:
        x0_out, x1_out = x0_in, x1_in
        y0_out, y1_out = y0_in, y1_in
    else:
        x_out, y_out = utl.prjxy( prjcode_raster, prjcode_out, 
                                  ( x0_in, x1_in ), ( y0_in, y1_in ) )
        x0_out, x1_out = x_out[0], x_out[1]
        y0_out, y1_out = y_out[0], y_out[1]
        
    if m2km is True :
         x0_out, y0_out, x1_out, y1_out =  x0_out/1e3, y0_out/1e3, x1_out/1e3, y1_out/1e3

    if close == True: 
        path_name = raster.GetDescription()
        raster = None
        raster = path_name

    if out == 'xlyl':
        return [x0_out, x1_out, y0_out, y1_out]
    if out == 'xyl':
        return [ x0_out, y0_out, x1_out, y1_out]

# -----------------------------------------------------------------------------
def raster_nodata( raster, band=1, close=False, p=False ):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    band = raster.GetRasterBand(band)
    nodata = band.GetNoDataValue()
    if p is True :
        print(nodata)

    if close == True:
        raster = None

    return nodata

# -----------------------------------------------------------------------------
def set_raster_nodata(raster,
                      nodataval = None,
                      greater_than = None,
                      less_than = None,
                      band = 1,
                      close = False,
                      path = None,
                      new_name = None,
                      copy = False,
                      plot = False,
                      new_fig = True):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    if copy is True:
        if new_name == None:
            new_name = raster_name(raster)+'_nodata'
        raster = raster_copy(raster, new_name=new_name, path=path)

    band = raster.GetRasterBand(band)
    array = band.ReadAsArray()

    if nodataval is not None :
        if utl.isiterable( nodataval ):
            ndv = nodataval[0]
            band.SetNoDataValue(ndv)
            for n in nodataval:
                array[array == n] = ndv
            band.WriteArray( array )
        else:
            band.SetNoDataValue(nodataval)

    ndv = band.GetNoDataValue()

    if greater_than is not None:
         array[(array>=greater_than) & (array!=ndv)] = ndv
         band.WriteArray( array )

    if less_than is not None:
         array[(array<=less_than) & (array!=ndv)] = ndv
         band.WriteArray( array )

    if plot == True:
        pltr(raster, new_fig=new_fig)

    if close == True:
        raster = None

    return raster

# -----------------------------------------------------------------------------
def raster_res(raster, out_prjcode=None, mean=False, close=False):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    if out_prjcode is None:
        gt = raster.GetGeoTransform()
    if out_prjcode is not None:
        rasterprj = raster_warp(raster, out_prjcode=out_prjcode)
        gt = rasterprj.GetGeoTransform()
    rx, ry = gt[1], -gt[5]

    if close == True:
        raster = None

    if mean == False:
        return rx, ry

    if mean == True:
        return np.mean( ( rx, ry ) )

# -----------------------------------------------------------------------------
def raster_prjcode( raster, close=False ) :
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    code = osr.SpatialReference(wkt=raster.GetProjection())
    out_prjcode = utl.prj_(code.ExportToProj4()).srs    

    if close == True:
        raster = None
        
    return out_prjcode    

# -----------------------------------------------------------------------------
def raster_epsg( raster, close=False ):
    """
    Get EPSG (reference system code) of a gdal_raster file
    """
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True
        
#    prjcode = raster_prjcode( raster )
#    epsg = utl.prj2epsg( prjcode )
        
    proj = osr.SpatialReference( wkt=raster.GetProjection() )
    epsg = int(proj.GetAttrValue('AUTHORITY',1))

    if close == True:
        raster = None

    return epsg

# -----------------------------------------------------------------------------
def raster_path(raster, close=False):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    path_name_extension = raster.GetDescription()
    path = os.path.dirname(os.path.realpath(path_name_extension))

    if close == True:
        # path_name = raster.GetDescription()
        raster = None

    return path

# -----------------------------------------------------------------------------
def raster_extension(raster, close=False):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    path_name_extension = raster.GetDescription()
    extension = path_name_extension.split( '.' )[-1]

    if close == True:
        # path_name = raster.GetDescription()
        raster = None

    return extension

# -----------------------------------------------------------------------------
def raster_description( raster, close=False ):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    Description = raster.GetDescription()

    if close == True:
        # path_name = raster.GetDescription()
        raster = None

    return Description

# -----------------------------------------------------------------------------
def raster_name(raster, close=False, extension=False, p=False):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    path_name_extension = raster.GetDescription()
    name_extension = path_name_extension.replace('/',os.sep).split(os.sep)[-1]
    if extension == False:
        name = name_extension.split('.')[0]
    if extension ==  True:
        name = name_extension

    if close == True:
        # path_name = raster.GetDescription()
        raster = None

    return name

# -----------------------------------------------------------------------------
def pltr( raster, band=1, vmin=None, vmax=None, tit=None, lim=True, stat=6, sbplt=[],
          cmap='rainbow', axis=False, new_fig=True, contours=[], adjst_lim=True,
          flipud=False, hillshade=False, ve=2, aspect='auto', blend_mode='overlay',
          mask=None, points=None, pc='k', ps=1, label=None, label_size='large',
          xlabel=None, ylabel=None, x_ax=True, y_ax=True, letter=None, 
          xlett=0, ylett=0, colorbar=True, rgb=False, close=True, print_stat=True,
          alpha=1, lines=None, lc='k', lett_size='large', cc=None, lett_colour='k',
          xRes=None, yRes=None, method=gdal.GRA_Bilinear, m2km=False ):

    if type(raster) == str:
        raster = gdal.Open( raster )
        close = True
        
    if type( lim ) in ( list, tuple ) : 
        raster = raster_warp( raster, lim, new_name=raster_name(raster) + '_wrap',
                              close=False, xRes=xRes, yRes=yRes, method=method ) 
    if lim is True :
        lim = raster_lim( raster )        
        
    if lim is False : 
        lim = None        

    if rgb is False :
        rband = raster.GetRasterBand( band )
        nodata = rband.GetNoDataValue()
        array = rband.ReadAsArray().astype( float )
        array[ array == nodata ] = np.nan
        
    if rgb is True :
        array = RGBraster2array( raster )[0]
        colorbar = False
        cmap = None
        
    if m2km == True :
        lim = [ i / 1e3 for i in lim ]
             
    ax = utl.plta( array, vmin=vmin, vmax=vmax, tit=tit, lim=lim, stat=stat, sbplt=sbplt,
              cmap=cmap, axis=axis, new_fig=new_fig, contours=contours, adjst_lim=adjst_lim,
              flipud=flipud, hillshade=hillshade, ve=ve, aspect=aspect, blend_mode=blend_mode,
              mask=mask, points=points, pc=pc, ps=ps, label=label, label_size=label_size,
              xlabel=xlabel, ylabel=ylabel, x_ax=x_ax, y_ax=y_ax, letter=letter, 
              xlett=xlett, ylett=ylett, colorbar=colorbar, print_stat=print_stat,
              alpha=alpha, lines=lines, lc=lc, lett_size=lett_size, cc=cc, lett_colour=lett_colour )

    if close == True:
        raster = None

    return ax

# -----------------------------------------------------------------------------
def raster_warp( raster,
                 lim = None,
                 xRes = None,
                 yRes = None,
                 width = None,
                 height = None,
                 lim_prjcode =None,
                 out_prjcode = None,
                 new_path = '/vsimem/',
                 new_name = None,
                 srcNodata = None,
                 dstNodata = None,
                 close = False,
                 extension = 'vrt',
                 method = gdal.GRA_Average,
                 cutlineDSName = None,
                 cropToCutline = False,
                 tps = False,
                 rpc = False,
                 geoloc = False,
                 errorThreshold = 0,
                 options = [],
                 plot = False,
                 vmin = None,
                 vmax = None) :
    
    """
    Crop raster to the defined bundaries --> latlon = [ minLon, maxLon, minLat, maxLat ]
    ReSample raste to the defined x,y grid_steps --> xRes, yRes.
    ReProject raster to the defined
    Using specific gdal interpolation method --> method
    """

    if type(raster) == str:
        raster = gdal.Open(raster)

    # Set destination path/name
    if new_path is None:
        new_path = raster_path(raster) + os.sep

    if new_name is None:
        new_name = raster_name(raster)

    if extension is None:
        extension = raster_extension(raster)

    if new_path != '/vsimem/':
        os.makedirs(new_path, exist_ok=True)
        new_path = new_path +os.sep

    dest_path_name = new_path + new_name + '.' + extension

    # Set output ReferenceSystem
    if out_prjcode is None:
        code = osr.SpatialReference(wkt=raster.GetProjection())
        out_prjcode = utl.prj_(code.ExportToProj4()).srs
    else:
        out_prjcode= utl.prj_(out_prjcode).srs

    # Set limits of the inout raster
    if lim_prjcode is not None:
        lim_prjcode = utl.prj_(lim_prjcode)

    if lim is not None:
        lim = utl.lim_sort(lim)

    if cutlineDSName is not None :
        shp_prjcode = shp.shp_prjcode( cutlineDSName )
        gdal.SetConfigOption('GDALWARP_IGNORE_BAD_CUTLINE','TRUE')
        if shp_prjcode != out_prjcode :
            cutlineDSName = shp.shp_translate( cutlineDSName, out_prjcode=out_prjcode )

    if ( width != None ) and ( height != None ) :
        if lim is None :
            lim = raster_lim( raster, out='xyl' )
        xRes = ( lim[2] - lim[0] ) / width
        yRes = ( lim[3] - lim[1] ) / height
        
#    if xRes is None :
#        xRes = raster_res( raster )[0]

#    if yRes is None :
#        yRes = raster_res( raster )[1]        
    # Crop reSample and reProject input raster (Warp)
    raster_out=gdal.Warp(destNameOrDestDS = dest_path_name, 
                         srcDSOrSrcDSTab = raster,
                         dstSRS = out_prjcode,
                         xRes = xRes,
                         yRes = yRes,
                         outputBounds = lim,
                         outputBoundsSRS = lim_prjcode,
                         srcNodata = srcNodata,
                         dstNodata = dstNodata,
                         resampleAlg = method,
                         cutlineDSName = cutlineDSName,
                         errorThreshold = errorThreshold,
                         tps = tps,
                         rpc = rpc,
                         geoloc = geoloc,                         
                         options = options )

    # Remove shp dataset (if used)
    if cutlineDSName is not None :
        cutlineDSName = None

    # Plot
    if plot==True:
        pltr(raster_out, vmin=vmin, vmax=vmax)

    if close == True:
        # gdal.Unlink( raster_out.GetDescription() )
        raster_out = None
        
        return dest_path_name
    
    else:
        return raster_out

# -----------------------------------------------------------------------------
def warp2ref( raster, raster_ref, method=gdal.GRA_Bilinear, 
              new_name=None, new_path='/vsimem/', extension='vrt', close=True,
              plot=False, vmin=None, vmax=None, epsg=False) :
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        
    lim = raster_lim( raster_ref ) 
    if epsg is True :
        out_prjcode = raster_epsg( raster_ref )
    else :    
        out_prjcode = raster_prjcode( raster_ref ) 
    lim_prjcode = out_prjcode
    height,  width = raster_shape( raster_ref ) 

    raster = raster_warp( raster,
                          lim = lim,
                          width = width,
                          height = height,
                          lim_prjcode =lim_prjcode,
                          out_prjcode = out_prjcode,
                          new_path = new_path,
                          new_name = new_name,
                          extension=extension,
                          method = method,
                          close = False) 
    
    if plot is True :
        pltr( raster, vmin=vmin, vmax=vmax )
    
    if close is True :
        path_name = raster.GetDescription()
        gdal.Unlink( path_name )
        raster = None
        raster = path_name
        
    return raster 

# -----------------------------------------------------------------------------
def raster_save( raster, new_path, new_name, extension='tif', close=True,
                 driver='GTiff'):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    os.makedirs( new_path, exist_ok = True )
    dst_filename = new_path  + os.sep + new_name + '.' + extension

    if os.path.isfile( dst_filename ) :
        os.remove( dst_filename )

    driver = gdal.GetDriverByName( 'GTiff' )
    new_raster = driver.CreateCopy( dst_filename, raster, 0 )

    if close==True:
        path_name = new_raster.GetDescription()
        new_raster = None
        raster = None 
        new_raster = path_name

    return new_raster

# -----------------------------------------------------------------------------
def resample_filt( raster, resampling_factor, overwrite=True, downsampling=gdal.GRA_Average,
                   upsampling=gdal.GRA_Bilinear, plot=False, vmin=None, vmax=None,
                   stat=5, close=True, new_name='raster_filt', path=None, suffix='' ):

    if overwrite is False :   
        raster = raster_copy( raster, new_name=new_name, extension='tif', suffix=suffix,
                              path=path, driver='GTiff', dtype=gdal.GDT_Float32 )   
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True   
    
    array0 = raster.GetRasterBand(1).ReadAsArray().astype(float)

    nodata = raster.GetRasterBand(1).GetNoDataValue()
    nan = array0 == nodata
    array0[nan] = np.nan

    dx, dy = raster_res( raster )
    width, height = raster_shape( raster )
    lim = raster_lim( raster )

    # Downsampling
    down = raster_warp( raster,
                        xRes = dx*resampling_factor,
                        yRes = dy*resampling_factor,
                        extension = 'tif',
                        new_name = 'down_filt',
                        method = downsampling )   
    
    array1 = raster2array( down )[2]
    array1[ array1 == nodata ] = np.nan

    # Upsampling
    up = raster_warp( down,
                      lim = lim,   
                      xRes = dx,
                      yRes = dy,
                      new_name = 'up_filt',
                      extension = 'tif',
                      method = upsampling )

    array2 = raster2array( up )[2]
    array2[nan] = nodata
    up.GetRasterBand(1).WriteArray(array2)
    array2[ (nan) | (array2==nodata) ] = np.nan

    raster.GetRasterBand(1).WriteArray( up.GetRasterBand(1).ReadAsArray() )    

    if plot==True:
        utl.plta(array0, tit=2, stat=stat, vmin=vmin, vmax=vmax, sbplt=[2,2,1] )
        utl.plta(array1, tit=2, stat=stat, vmin=vmin, vmax=vmax, sbplt=[2,2,2] )
        utl.plta(array2-array0, tit=2, stat=stat, sbplt=[2,2,3] )
        utl.plta(array2, tit=2, stat=stat, vmin=None, vmax=None, sbplt=[2,2,4] )
        
    down = None
    up = None
    
    if close==True:
        path_name = raster.GetDescription()
        raster = None
        raster = path_name    

    return raster

# -----------------------------------------------------------------------------
def array2raster( array, lonmin=0, latmax=0, rx=1, ry=1, reference_raster=None,
                  new_name='new_raster', path='/vsimem/', extension='tif', 
                  nodata=9999, new_folder=False, close=True, adjst_lim=True,
                  resampling=False, factor=2, plot=False, vmin=None, vmax=None, 
                  options=[''], eType=gdal.GDT_Float32, prjcode=4326 ) :
    
    if type( array ) in ( tuple, list ) :
        xx, yy, zz = array
        lonmin = xx.min()
        latmax = yy.max()
        rx, ry, _ = utl.stepxy( xx, yy )
    else :
        zz = array
    
    if resampling is True :
        if type( array ) in ( tuple, list ) :
            xx, yy, zz = utl.resampling( array, factor )
            rx, ry, _ = utl.stepxy( xx, yy ) 
        else :
            zz = utl.resampling( array, factor ) 
            rx = rx / factor
            ry = ry / factor                                     
    
    zz_new = np.copy( zz )
    zz_new[ np.isnan(zz) ] = nodata
    
    if reference_raster is not None :
        if type( reference_raster ) == str :
            reference_raster = gdal.Open( reference_raster )
        gt =  reference_raster.GetGeoTransform()
        rx = gt[1]
        ry = -gt[5]
        lonmin = gt[0]
        latmax = gt[3]
        prjcode = raster_epsg( reference_raster )
        
    if ( reference_raster is None ) and ( adjst_lim is True ) :
        lonmin = lonmin - rx / 2
        latmax = latmax + ry / 2

    if path != '/vsimem/':
        if new_folder == True :
            path = path + os.sep + new_name
            os.makedirs(path, exist_ok=True)
        path = path + os.sep
            
    rst_path_name = path + new_name + '.' + extension
    driver = gdal.GetDriverByName('GTiff')
    
    Raster_out = driver.Create( rst_path_name, zz.shape[1], zz.shape[0],
                                bands=1, eType=eType, options=options )

    Raster_out.SetGeoTransform([lonmin, rx, 0.0, latmax, 0.0, -ry])
    Raster_out.GetRasterBand(1).SetNoDataValue(nodata)
#    srs = osr.SpatialReference()
#    srs.ImportFromEPSG(prjcode)
#    srs.ImportFromWkt( utl.prj_(prjcode).to_wkt() )
#    print( utl.prj_(prjcode).to_wkt() )
    Raster_out.SetProjection( utl.prj_(prjcode).to_wkt( version='WKT1_GDAL') )
    out_band = Raster_out.GetRasterBand(1)
    out_band.WriteArray( zz_new )
    out_band.FlushCache()

    if reference_raster != None:
        reference_raster = None   
        
    if plot == True :        
        pltr( Raster_out, vmin=vmin, vmax=vmax )        

    if close==True:
        path_name = Raster_out.GetDescription()
        Raster_out = None
        Raster_out = path_name
        
    return Raster_out   
    
# -----------------------------------------------------------------------------
def RemRes( list_raster,
            lim = None,
            res = None,
            nodata = 9999,
            prjcode = 4326,
            lim_prjcode = None,
            shp_ply = None,
            InOut_ply = None,
            downsampling = gdal.GRA_Average,
            upsampling = gdal.GRA_CubicSpline,
            new_name = None,
            path = None,
            fillnan = False,
            maxSearchDist = 100,
            MinMax = None,
            smoothingIterations = 3,
            errorThreshold = None,
            plot = False,
            vmin = None,
            vmax = None ):

    """
    Remove Restore method to merge raster grids with different resolutions
    """

    if ( lim_prjcode is None ) or ( lim is None ) :
        lim_prjcode = prjcode

    # -------------------------------------------------------------------------
    # Reproject all rasters in the same reference system ----------------------
    print ('Re_project rasters to the same crs \n ...')

    r_rasters = [] # list with projected rasters
    res_list = [] # list with rasters resolution

    for i, r in enumerate( list_raster ):

        r_rasters.append( raster_warp( r,
                                       out_prjcode = prjcode,
                                       method = downsampling,
                                       dstNodata = nodata,
                                       extension = 'vrt' ) )

        res_list.append( raster_res( r_rasters[i], mean=True ) )

    print ('Done !!')
    #--------------------------------------------------------------------------
    # If no reslotion bounds are specified in input (i.e. res is None)
    # then, res is set equal to the minimum resolution found among the imput rasters

    if res is None:
        res = res_list.min()

    # -------------------------------------------------------------------------
    # Set map limits ----------------------------------------------------------
    if lim is None:
        xminl, xmaxl, yminl, ymaxl = [], [], [], []
        for r in r_rasters:
            r_lim = raster_lim(r, out='xlyl')
            xminl.append( r_lim[0] )
            xmaxl.append( r_lim[1] )
            yminl.append( r_lim[2] )
            ymaxl.append( r_lim[3] )
        lim = ( np.min( xminl ), np.max( xmaxl ), np.min( yminl ), np.max( ymaxl ))

    # -------------------------------------------------------------------------
    # Crop 2 limits -----------------------------------------------------------
    print ('Crop raster to limits \n ...')

    for i, r in enumerate( r_rasters ):

        if ( res is not None ) and ( raster_res( r, mean=True ) < res ):
            xRes = res
            yRes = res
            res_list[i] = res

        else:
            xRes = None
            yRes = None

        r_rasters[i] = raster_warp( r,
                                         lim = lim,
                                         lim_prjcode = lim_prjcode,
                                         xRes = xRes,
                                         yRes = yRes,
                                         method = downsampling,
                                         dstNodata = nodata,
                                         errorThreshold = errorThreshold,
                                         extension = 'vrt' )

    print ('Done !!')
    # -------------------------------------------------------------------------
    # Resample using the Min resolution selected
    # If a shapefile polygon is used, then all values inside or eventually ouside
    # the polygon are masked (InOut_ply, specify if True values for each raster
    # should be inside (1), or outside (0))
    print ('Resample rasters to minimum resolution \n ...')

    h_rasters = []
    if shp_ply is not None:

        # Cut shp to limits
        print ('    Set .shp polygon \n     ...')
        shp_prjcode = shp.shp_prjcode(shp_ply)
        shp_lim = utl.prj_lim(lim, lim_prjcode, shp_prjcode, sort='xyl')
        arguments = f'-clipsrc {shp_lim[0]} {shp_lim[1]} {shp_lim[2]} {shp_lim[3]}'
        if (new_name is not None) and (path is not None):
            shp_path = path + os.sep + new_name
        else:
            shp_path = None
        shp_cilp = shp.ogr2ogr(shp_ply, arguments, new_path=shp_path, suffix='_clip')

        shp_ds = ogr.Open( shp_cilp )
        lyr = shp_ds.GetLayer()
        print ('    Done !!')

        # if InOut_ply is None, than use 1 (inside) for each raster
        if InOut_ply is None:
            InOut_ply = [ 1 for i in len( h_rasters ) ]

        print ('    Set mask_array from .shp polygon \n    ...')
        temp_copy = raster_warp( r_rasters[0],
                                      lim = lim,
                                      xRes = res,
                                      yRes = res,
                                      lim_prjcode = lim_prjcode,
                                      dstNodata = 1,
                                      method = downsampling,
                                      new_name = 'temp',
                                      extension = 'tif')

        gdal.RasterizeLayer( temp_copy, [1], lyr, burn_values = [nodata] )
        # Get index of pixels that fall outside the .shp polygon (RNaN)
        PlyNaN = temp_copy.GetRasterBand(1).ReadAsArray() == nodata
        temp_copy = None # remove temporary dataset
        shp_ds = None # remove shp dataset
        lyr = None # remove shp layer
        print ('    Done !!')

    # Resample each raster to the minimum res.
    print ('    Completing resampling \n    ...')
    for i, r in enumerate(r_rasters):

        if res_list[i] >= res:
            method = upsampling

        if res_list[i] <= res:
            method = downsampling

        h_rasters.append( raster_warp( r,
                                            lim = lim,
                                            xRes = res,
                                            yRes = res,
                                            lim_prjcode = lim_prjcode,
                                            dstNodata = nodata,
                                            method = method,
                                            errorThreshold = errorThreshold,
                                            new_name = 'h'+str(i),
                                            extension = 'vrt' ) )

    print ('    Done !!')
    print ('Done !!')
    # -------------------------------------------------------------------------
    # Sort raster in h_raster list according to the original reolution,
    # starting from the grid with the larger grid_step (i.e. h_rasters[0])
    h_rasters = [ v[0] for v in sorted( zip( h_rasters, res_list ), key=lambda v: -v[1] ) ]
    res_list = [ v[1] for v in sorted( zip( h_rasters, res_list ), key = lambda v : -v[1] ) ]
    r_rasters = [ v[0] for v in sorted( zip( r_rasters, res_list ), key = lambda v : -v[1] ) ]
    # -------------------------------------------------------------------------
    print ('Completing Remove-Restore procedure \n ...')

    for i, r in enumerate( h_rasters ):
        if i == len(h_rasters) - 1:
            break

        Rg1 = raster_copy( h_rasters[i], new_name = 'Rg1', extension = 'tif' )
        Rg2 = raster_copy( h_rasters[i+1], new_name = 'Rg2', extension = 'tif' )

        R1 = h_rasters[i].GetRasterBand(1).ReadAsArray()
        R2 = h_rasters[i+1].GetRasterBand(1).ReadAsArray()

        if i == 0 :
            if shp_ply is not None :
                if InOut_ply[ i ] == 1 :
                    R1[ ~PlyNaN ] = nodata
                if InOut_ply[ i ] == 0 :
                    R1[ PlyNaN ] = nodata
                if InOut_ply[ i+1 ] == 1 :
                    R2[ ~PlyNaN ] = nodata
                if InOut_ply[i+1] == 0 :
                    R2[ PlyNaN ] = nodata
        if i != 0 :
            if shp_ply is not None :
                if InOut_ply[ i+1 ] == 1 :
                    R2[ ~PlyNaN ] = nodata
                if InOut_ply[ i+1 ] == 0 :
                    R2[ PlyNaN ] = nodata

        nanR1 = R1 == nodata
        nanR2 = R2 == nodata
        R1[nanR1] = R2[nanR1]
        R2[nanR2] = R1[nanR2]

        # Remove --------------------------------------------------------------
        Rem = R2 - R1
        Rem[nanR1 & nanR2] = nodata
        # ---------------------------------------------------------------------
        utl.plta(Rem, new_fig=True, tit='Rem')

        Rg1.GetRasterBand(1).WriteArray( Rem )

        Rg1 = raster_warp( Rg1,
                                xRes = res_list[i],
                                yRes = res_list[i],
                                method = downsampling,
                                errorThreshold = errorThreshold,
                                lim = lim,
                                lim_prjcode = lim_prjcode,
                                dstNodata = nodata,
                                extension = 'tif' )

        Rg2 = raster_warp( Rg2,
                                xRes = res_list[i],
                                yRes = res_list[i],
                                method = downsampling,
                                errorThreshold = errorThreshold,
                                lim = lim,
                                lim_prjcode = lim_prjcode,
                                dstNodata = nodata,
                                extension = 'tif' )

        Rgi = raster_warp( h_rasters[i],
                                xRes = res_list[i],
                                yRes = res_list[i],
                                method = downsampling,
                                errorThreshold = errorThreshold,
                                lim = lim,
                                lim_prjcode = lim_prjcode,
                                dstNodata = nodata,
                                extension = 'vrt',
                                new_name = 'Rgi')


        A1 = Rg1.GetRasterBand(1).ReadAsArray()
        A2 = Rg2.GetRasterBand(1).ReadAsArray()

        nanA1 = A1 == nodata
        nanA2 = A2 == nodata
        A2[nanA2] = Rgi.GetRasterBand(1).ReadAsArray()[nanA2]

        nan3 = ( A2 == nodata ) & nanA1

        # Restore -------------------------------------------------------------
        Res = A1 + A2
        Res[nan3] = nodata
        utl.plta(Res, new_fig=True, tit='res')
        # ---------------------------------------------------------------------

        Rg3 = raster_copy( Rg2, new_name = 'Rg3', extension = 'tif' )
        Rg3.GetRasterBand(1).WriteArray( Res )

        h_rasters[i+1] = raster_warp( Rg3,
                                            xRes = res,
                                            yRes = res,
                                            method = gdal.GRA_Cubic,
                                            errorThreshold = errorThreshold,
                                            lim = lim,
                                            lim_prjcode = lim_prjcode,
                                            dstNodata = nodata,
                                            extension = 'tif',
                                            new_name = raster_name( h_rasters[i+1] ) )

        C = h_rasters[i+1].GetRasterBand(1).ReadAsArray()
        C[nanR1 & nanR2] = nodata
        if MinMax is not None:
            C[ ( C<=MinMax[0] ) | ( C>=MinMax[1] ) ] = nodata
        h_rasters[i+1].GetRasterBand(1).WriteArray(C)

        del Rg1, Rg2, Rg3, Rgi

    ResFinal = h_rasters[-1]

    # Fill remaining nodata gaps ----------------------------------------------
    if fillnan==True:
        gdal.FillNodata( targetBand = ResFinal.GetRasterBand(1),
                          maskBand = None,
                          maxSearchDist = maxSearchDist,
                          smoothingIterations = smoothingIterations )

    if new_name is not None:
        ResFinal = raster_copy( ResFinal, new_name=new_name, path=path )

    # Plot resoult with Matplot lib (NOT suggested if the file is too large) --
    if plot==True:
        pltr( ResFinal, vmin=vmin, vmax=vmax )

    print ('Done !!')
    # -------------------------------------------------------------------------

    # Remove final raster from temp. Memory -----------------------------------

    path_name = ResFinal.GetDescription()
    gdal.Unlink( ResFinal.GetDescription() )
    ResFinal = None
    ResFinal = path_name

    for i, r in enumerate( h_rasters ):
        del r_rasters[i], h_rasters[i]

    shp_ds = None

    return ResFinal

# -----------------------------------------------------------------------------
def raster_merge( raster_list,
                  sigmab = 'auto',
                  new_name = 'raster_merged',
                  path = '/vsimem/',
                  nodata = 9999,
                  close = False,
                  plot = False,
                  vmin = None,
                  vmax = None ) :
    
    for i, r in enumerate( raster_list ) :
        if type( r ) == str:
           raster_list[ i ] = gdal.Open( r ) 
    
    Z1 =  raster_list[0].GetRasterBand(1).ReadAsArray().astype(float)
    Z1[ Z1 == nodata ] = np.nan

    for i, r in enumerate( raster_list ) :
        
        if i == 0 : continue
        
        Z2 =  r.GetRasterBand(1).ReadAsArray().astype(float)
        Z2[ Z2 == nodata ] = np.nan
        shape1 = Z1.shape
        shape2 = Z2.shape
        shape_ratio = shape1[0]/shape2[0], shape1[1]/shape2[1]
        res_ratio = round( raster_res( r, mean=True ) / 
                           raster_res( raster_list[i-1], mean=True ) )    
        if sigmab == 'auto' :
            sigma = int( res_ratio )
        else :
            sigma = sigmab
            
        print( 'sigmab: ', sigma )
        Z21 = ndimage.zoom( Z2, shape_ratio, order=1 )    
        
        mask1 = np.isfinite( Z1 )
        mask2 = np.isfinite( Z21 )
        mask1n2 = mask1 & ~mask2
    
        # ---
        # If the i-th raster has the original resolution lower than the final chosen resolution,
        # it will be smoothed with a moving average convolution, 
        # with a kernel size equal to the ratio betwee original and final resolution (+1 if it's even).
        # This will reduce aliasing artefacts in the low resolution area  
        if res_ratio > 1 :
            Z21[ ~mask2 ] = 0 
            if res_ratio % 2 == 0 :
                Z21 = ndimage.uniform_filter( Z21 , res_ratio+1 )
            else :
                Z21 = ndimage.uniform_filter( Z21 , res_ratio )
            Z21[ ~mask2 ] = np.nan 
        # ---  
        
        D_21 = Z1 - Z21  
        mask1a2 = np.isfinite( D_21 )  
    
        # If there are no common pixel between the two arrays, 
        # they will be merged as they are withut blending the edges 
        # ---
        if np.all( mask1a2 == False  ):
            Z1[mask2] = Z21[mask2]
            continue        
        
        Z1[ ~mask1 ] = 0
        Z21[ ~mask2 ] = 0
        D_21[ ~mask1a2 ] = np.nan
        D_21[ mask1n2 ] = Z1[ mask1n2 ]
        
        mask3 = ~ndimage.gaussian_filter( ~mask1*1, sigma*2 ).astype(bool)
        mask4 = ~ndimage.gaussian_filter( ~mask1*1, sigma ).astype(bool)
        mask5 = ndimage.uniform_filter( mask1*1, 6 ).astype(bool)  
        
        weight = ndimage.gaussian_filter( mask4*1.0, sigma*2) 
        weight[ mask1n2 ] = 1
        D_21[ ~mask2  ] = np.nan
        
        D_21fn = utl.grid_fill( D_21, invalid=None )
        D_21fn[ ~mask3  ] = 0
        
        FG = np.copy( D_21fn )
        sign = np.sign( FG )
        FG = ndimage.maximum_filter( np.abs(FG) , 3 )*sign
        FG[ mask1 ] = D_21fn[ mask1 ]
        
        for i in range( int( sigma*2 ) ) :   
            FG1 = ndimage.uniform_filter( FG , 3 )
            FG1[ mask5 ] = D_21fn[ mask5 ]
            FG = ndimage.uniform_filter( FG1 , 3 )
            FG[ mask4 ] = FG1[ mask4 ]
            FG[ mask1n2 ] = D_21fn[ mask1n2 ]
        
        FG = FG * weight 
        FG[ mask1n2 ]  = Z1[ mask1n2 ] 
        
        Z2n = Z21 + FG
        Z2n[ ~( mask1 | mask2 ) ] = np.nan  
        Z1 = Z2n
        
    rm = array2raster( Z1, reference_raster=raster_list[0], new_name=new_name,
                     path=path, extension='tif', close=False )  
      
    if plot is True :
        pltr( rm, vmin=vmin, vmax=vmax)
        
    if close is True :
        path_name = rm.GetDescription()
        rm = None
        rm = path_name
        
    return rm    

# -----------------------------------------------------------------------------
def RGBraster2array( RGB_geotif, close=False ):
    """
    Authors: Bridget Hass
    Ref: https://www.neonscience.org/resources/learning-hub/tutorials/plot-neon-rgb-py
    Last Updated: Oct 7, 2020

    -------
    RGBraster2array reads in a NEON AOP geotif file and returns
    a numpy array, and header containing associated metadata with spatial information.
    --------
    Parameters
        RGB_geotif -- full or relative path and name of reflectance hdf5 file
    --------
    Returns
    --------
    array:
        numpy array of geotif values
    metadata:
        dictionary containing the following metadata (all strings):
            array_rows
            array_cols
            bands
            driver
            projection
            geotransform
            pixelWidth
            pixelHeight
            extent
            noDataValue
            scaleFactor
    --------
    Example Execution:
    --------
    RGB_geotif = '2017_SERC_2_368000_4306000_image.tif'
    RGBcam_array, RGBcam_metadata = RGBraster2array(RGB_geotif)
    """

    metadata = {}
    if type(RGB_geotif) == str: dataset = gdal.Open(RGB_geotif)
    else: dataset = RGB_geotif

    metadata['array_rows'] = dataset.RasterYSize
    metadata['array_cols'] = dataset.RasterXSize
    metadata['bands'] = dataset.RasterCount
    metadata['driver'] = dataset.GetDriver().LongName
    metadata['projection'] = dataset.GetProjection()
    metadata['geotransform'] = dataset.GetGeoTransform()

    metadata['extent'] = raster_lim(dataset)
    metadata['xy_array'] = raster2array(dataset)[0:2]

    raster = dataset.GetRasterBand(1)
    array_shape = raster.ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype(float).shape
    metadata['noDataValue'] = raster.GetNoDataValue()

    RgbArray = np.zeros((array_shape[0],array_shape[1],dataset.RasterCount),'uint8') #pre-allocate stackedArray matrix
    for i in range(1, dataset.RasterCount+1):
        band = dataset.GetRasterBand(i).ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype(float)
        band[band==metadata['noDataValue']]=np.nan
        RgbArray[...,i-1] = band

    dataset = None
    

    return RgbArray, metadata

# -----------------------------------------------------------------------------
def raster_shape( raster, close=False ):

    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True

    width = raster.RasterXSize
    height = raster.RasterYSize

    if close == True :
        raster = None

    return height, width

# -----------------------------------------------------------------------------
def raster_list_lim( raster_list,
                     out='xlyl',
                     method = 'max',
                     prjcode_out=None,
                     midpx=False,
                     close=False ) :

    xmin, xmax, ymin, ymax = [], [], [], []
    
    for r in raster_list:
        r_lim = raster_lim(r, out='xlyl', prjcode_out=prjcode_out, midpx=midpx )
        xmin.append( r_lim[0] )
        xmax.append( r_lim[1] )
        ymin.append( r_lim[2] )
        ymax.append( r_lim[3] )
    if method == 'max' :
        MinMaxLim = ( np.min( xmin ), np.max( xmax ), np.min( ymin ), np.max( ymax ) )
    if method == 'min' :
        MinMaxLim = ( np.max( xmin ), np.min( xmax ), np.max( ymin ), np.min( ymax ) )
    if out != 'xlyl' :
        MinMaxLim = utl.lim_sort( MinMaxLim )

    return MinMaxLim

# -----------------------------------------------------------------------------
def raster_sort( raster_list,
                 lim = None,
                 min_res = None,
                 nodata = 9999,
                 prjcode = 4326,
                 lim_prjcode = None,
                 int_attribute = [],
                 method = 'up',
                 downsampling = gdal.GRA_Average,
                 crop2lim = True,
                 extension = 'vrt',
                 output = 'raster_list' ) :

    gdal.VSICurlClearCache() 
    new_list = copy( raster_list )
    """
    Sort list of raster files based on thier resolution and attributes
    """
    if ( lim_prjcode is None ) or ( lim is None ) :
        lim_prjcode = prjcode

    # -------------------------------------------------------------------------
    # (1) Reproject all rasters to the same reference system( i.e. "prjcode" )
    # (2) Append all rasters's mean grid-steps to a list ( i.e. "res_list" )

    res_list = [] # list with rasters resolution
    
    for i, r in enumerate( raster_list ):
        if type( raster_list[i] ) == str :
            new_list[i] = gdal.Open( raster_list[i] )
        if raster_prjcode( r ) != utl.prj_( prjcode ).srs :
            new_list[i] = raster_warp( r,
                                       out_prjcode = prjcode,                                      
                                       method = gdal.GRA_Bilinear,
                                       dstNodata = nodata,
                                       extension = extension,
                                       new_name = raster_name(r)+'_r'+str(i) )

        res_list.append( raster_res( new_list[i], mean=True ) )  
    #--------------------------------------------------------------------------
    # (3) If no final reslotion in specified in the inputs (i.e. res is None)
    #     then, res is set equal to the minimum resolution found among the input rasters
    #     (i.e. within "res_list")

    if min_res is None:
        min_res = np.min( res_list )

    # -------------------------------------------------------------------------
    # (4) Set map limits:
    #     if no limits have been specifies in the inputs (i.e. lim is None)
    #     then, lim are set equal to the minimum/maximum limits found among the imput rasters
    if lim is None:
        lim = raster_list_lim( raster_list, out='xlyl', method = 'max' )

    # -------------------------------------------------------------------------
    # (5) Crop to limits:
    #     crop all raster to the limits specified in the variable lim = ( xmin, xmax, ymin, ymax ).
    #     If some rasters have a mean grid-step smaller than the final selected grid-step (i.e. "res"),
    #     they will be downsampled to --> grid-step = res

    if crop2lim is True :
        for i, r in enumerate( raster_list  ):
            if res_list[i] < min_res :
                xRes = min_res
                yRes = min_res
                
            else:
                xRes = None
                yRes = None
              
            new_list[i] = raster_warp( r,
                                          out_prjcode = prjcode, 
                                          lim = lim,
                                          lim_prjcode = lim_prjcode,
                                          xRes = xRes,
                                          yRes = yRes,
                                          method = downsampling,
                                          dstNodata = nodata,
                                          extension = extension,
                                          new_name = raster_name(r)+'_c'+str(i) )

    # -------------------------------------------------------------------------
    # (6) Define raster attribute if it was not defined in the inputs ( i.e. if "int_attribute=[]" )
    #     If not defined all rasters will have the same attribute = 1
    if int_attribute == [] :
        int_attribute = [ 1 for i in raster_list ]

    # -------------------------------------------------------------------------
    # Sort raster in raster_list list according to the original reolution,
    # starting from the grid with the smaller grid_step if method if "method" == "up",
    # or from the one with the larger grid-step if "method" == "down"
    # Then, the list is also sort accordinge to the ascending attribute values
    # (i.e. int_attribute)

    if method == 'up' :
        m = 1
    if method == 'down' :
        m = -1

    r_sort = [ [ v[0], v[1], v[2] ] for v in sorted( zip( new_list, res_list, int_attribute ),
               key=lambda v: ( v[2], m*v[1] ) ) ]
    
    if output == 'all' :
        return r_sort

    if output == 'raster_list' :
        sorted_raster_list = [ i[0] for i in r_sort ]
        return sorted_raster_list

# -----------------------------------------------------------------------------
def raster_write( raster, array, band=1, 
                  copy = False, 
                  new_name = 'copy',
                  extension = 'tif',
                  suffix = '',
                  path = None,                  
                  close = False, 
                  plot = False, vmin = None, vmax = None ) :
    

    if copy is True :
        raster = raster_copy( raster, new_name=new_name, suffix=suffix, 
                              path=path, extension=extension, close=False)     
    
    if type(raster) == str:
        raster = gdal.Open( raster )
    
    if raster_extension == 'vrt' :
        print( 'This operation is not allowed with ".vrt" files ( use raster-files ".tif" ) ' )
        path_name = raster.GetDescription()
        raster = None
        return

    nodata = raster_nodata( raster )
    nan = np.isnan( array )
    array[nan] = nodata
    raster.GetRasterBand( band ).WriteArray( array )
    
    if plot is True :
        pltr( raster, vmin=vmin, vmax=vmax )

    if close is True :
        path_name = raster.GetDescription()
        del raster
        raster = path_name
    
    return raster    

# -----------------------------------------------------------------------------
def raster_fill( raster, maxSearchDist=100, smoothingIterations=3, 
                 copy=False, path='/vsimem/', new_name=None, suffix='', extension='tif',            
                 close=True, band=1, out='raster', plot=False, vmin=None, vmax=None) :
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        close = True   

    if copy == True :
        if new_name is None :
            new_name = raster_name ( raster )
        raster = raster_copy( raster, new_name=new_name, path=path, suffix=suffix,
                              extension = extension)  
        
    raster_band = raster.GetRasterBand(band)   
        
    gdal.FillNodata( targetBand = raster_band, maskBand = None,
                     maxSearchDist = maxSearchDist, smoothingIterations = smoothingIterations)     

    if plot is True :
        pltr( raster, vmin=vmin, vmax=vmax )
    
    if out == 'array' :
        _,_,array = raster2array( raster, nodata=np.nan )
        return array 
    
    if close is True :
        path_name = raster.GetDescription()
        raster = None
        del raster
        raster = path_name
    
    return raster      
# -----------------------------------------------------------------------------
def rasterize( raster, shp_ply, path='/vsimem/', close=False, copy=True, new_name=None,
               suffix='_msk', inverse=False, nodata=None , fillnan=True,
               maxSearchDist=100, smoothingIterations=3, plot=False, vmin=None, vmax=None ) :

    shp_prjcode = shp.shp_prjcode( shp_ply ) 
    rst_prjcode = raster_prjcode( raster )
    lim = raster_lim( raster )
    lim_prj = raster_prjcode( raster )
    
    if shp_prjcode != rst_prjcode :
        lim_shp = utl.extend_lim( lim, 50, method='percentage')
        shp_ply = shp.translate( shp_ply, new_name=None, new_path='/vsimem/', 
                                 lim=lim_shp, lim_prjcode=lim_prj)    
    
    if type(raster) == str:
        raster = gdal.Open( raster )
        
    if type(shp_ply) == str:    
        shp_ply = ogr.Open( shp_ply )  
        
    lyr = shp_ply.GetLayer()  
    
    if copy == True :
        if new_name is None :
            new_name = raster_name ( raster )
        raster = raster_copy( raster, new_name=new_name, path=path, suffix=suffix,
                              extension = 'tif')
    if nodata is None :
        nodata = raster_nodata( raster )
        
    if fillnan is True :
        gdal.FillNodata( targetBand = raster.GetRasterBand(1), maskBand = None,
                         maxSearchDist = maxSearchDist,
                         smoothingIterations = smoothingIterations)         

    if inverse is False : 
        gdal.RasterizeLayer( raster, [1], lyr, burn_values=[ nodata ] )
    else :
#        mask = raster_copy( raster, new_name='mask', extension='tif')  
#        gdal.RasterizeLayer( mask, [1], lyr, burn_values=[ nodata ] )
        mask = raster_mask( raster, shp_ply, path_name=None, plot=False )
        mask_array = mask.GetRasterBand(1).ReadAsArray()
        revese_nan = mask_array != 0
        raster_array = raster.GetRasterBand(1).ReadAsArray()
        raster_array[ revese_nan ] = nodata
        raster.GetRasterBand(1).WriteArray( raster_array )

    if plot is True :
        pltr(raster, vmin=vmin, vmax=vmax)
    
    if close is True :
        path_name = raster.GetDescription()
        raster = None
        raster = path_name
        
    return raster       

# -----------------------------------------------------------------------------   
def merge( raster_list,
           lim = None,
           prjcode = 4326,
           lim_prjcode = None,
           min_res = None,
           nodata = 9999,
           int_attribute = [],
           downsampling = gdal.GRA_Average,
           method = 'merge',
           threshold = 50,
           shp_ply = None,
           sigmab = 1,
           name = 'raster_merged',
           path = None,
           plot = False,
           vmin = None,
           vmax = None,
           fillnan1 = True,
           fillnan2 = False,
           final_res = None,
           maxSearchDist = 100,
           smoothingIterations = 3,
           close = True ):
    
    if os.path.exists( path + os.sep + name + '.tif' ) :
        os.remove( path + os.sep + name + '.tif' )    
    
    print("Sorting raster files ...")
    lst_out = raster_sort( raster_list, 
                           lim = lim, 
                           min_res = min_res, 
                           nodata = nodata,
                           prjcode = prjcode, 
                           lim_prjcode = lim_prjcode, 
                           int_attribute = int_attribute,
                           downsampling = downsampling,
                           crop2lim = True,
                           extension = 'vrt',
                           output = 'all')
    
    raster_in = []
    raster_out = []
    res_in = []
    res_out = []
    
    for i, r in enumerate( lst_out ) :
        if r[2] == 1 :
            raster_in.append( r[0] )
            res_in.append( r[1] )
            print( raster_name(r[0]), r[1] , 1 )
        if r[2] == 0 :
            raster_out.append( r[0] ) 
            res_out.append( r[1] )
            print( raster_name(r[0]), r[1] , 0 )

    if method == 'merge' :
        if raster_in != [] :
            print ( "Merge rasters in (1) ...")
            if np.size ( raster_in ) > 1 :
                array_list = []
                for ri in raster_in :
                    array_list.append( ri.ReadAsArray() )
                    isnan = array_list[-1] == nodata
                    array_list[-1][isnan] = np.nan
                    path_name = ri.GetDescription()
                    
                a_in = utl.merge2Darrays( array_list, res_list=res_in, sigmab=sigmab )
                r_in = array2raster( a_in, reference_raster=raster_in[0], 
                            new_name='r_in', path=path, extension='tif', close=False )  
#                r_in = raster_merge( raster_in, sigmab=sigmab, new_name='r_in' )
            else :
                r_in = raster_in[0]
        
        if raster_out != [] :
            print ( "Merge rasters out (0) ...")
            if np.size ( raster_out ) > 1 :
                array_list = []
                for ri in raster_out :
                    array_list.append( ri.ReadAsArray() )
                    isnan = array_list[-1] == nodata
                    array_list[-1][isnan] = np.nan  
                    path_name = ri.GetDescription()
                    
                a_out = utl.merge2Darrays( array_list, res_list=res_out, sigmab=sigmab )
                r_out = array2raster( a_out, reference_raster=raster_out[0], 
                            new_name='r_out', path=path, extension='tif', close=False )                                 
#                r_out = raster_merge( raster_out, sigmab=sigmab, new_name='r_out' )
            else :
                r_out = raster_out[0]                 
 
    if shp_ply is not None : 
        if raster_in != [] :
            print ( "Mask rasters in polygon (shp_ply) ...")
            r_in = rasterize( r_in, shp_ply, inverse=True, fillnan=fillnan1,
                              maxSearchDist=maxSearchDist, smoothingIterations=smoothingIterations ) 
        
        if raster_out != [] :
            print ( "Mask rasters out polygon (shp_ply) ...")
            r_out = rasterize( r_out, shp_ply, fillnan=fillnan1,
                               maxSearchDist=maxSearchDist, smoothingIterations=smoothingIterations )  
      
    if ( raster_in != [] ) and ( raster_out != [] ):
        _,_,a_in = raster2array( r_in, nodata=np.nan, close = False )   
        _,_,a_out = raster2array( r_out, nodata=np.nan, close = False  )
        a_in[ np.isnan( a_in ) ] = a_out[ np.isnan( a_in ) ]
        a_in[ np.isnan( a_in ) ] = nodata
        raster = raster_write( r_in, a_in, copy=True, new_name=name, extension='tif', path=path, close=False )
    if ( raster_in != [] ) and ( raster_out == [] ):        
        raster = raster_copy( r_in, new_name=name, extension='tif', path=path, close=False )
    if ( raster_in == [] ) and ( raster_out != [] ):        
        raster = raster_copy( r_out, new_name=name, extension='tif', path=path, close=False )        
    
    if fillnan2 is True :
        print ( "Filling remaining nodata val ...")
        gdal.FillNodata( targetBand = raster.GetRasterBand(1),
                         maskBand = None,
                         maxSearchDist = maxSearchDist,
                         smoothingIterations = smoothingIterations) 
        
    if final_res != None :
        raster = raster_warp( raster, xRes=final_res, yRes=final_res, method=downsampling,
                              new_path=path, new_name=name, extension='tif')
    
    if raster_in != [] :
        for i in raster_in :
            path_name = i.GetDescription()
            i = None
    if raster_out != [] :    
        for i in raster_out :
            path_name = i.GetDescription()
            i = None
    shp_ply = None
    
    if plot is True :
        pltr( raster, vmin, vmax )
        
    if close is True :
        path_name = raster.GetDescription()
        raster = None
        raster = path_name

    return raster

# -----------------------------------------------------------------------------
def add_coastline_band( raster, ply_coast, new_path, new_name ) :
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        
    shp_prjcode = shp.shp_prjcode( ply_coast ) 
    rst_prjcode = raster_prjcode( raster )
    lim = raster_lim( raster )
    lim_prj = raster_prjcode( raster )
    if shp_prjcode != rst_prjcode :
        lim_shp = utl.extend_lim( lim, 50, method='percentage')
        shp_ply = shp.translate( ply_coast, new_name=None, new_path='/vsimem/', 
                                 lim=lim_shp, lim_prjcode=lim_prj)          
   
    if type(shp_ply) == str:    
        shp_ply = ogr.Open( shp_ply )  
    
    lyr = shp_ply.GetLayer()   
    tmp_mask = gdal.GetDriverByName('MEM').CreateCopy('', raster, 0)
    tmp_mask.AddBand()
    tmp_mask.GetRasterBand(2).Fill(1)
    gdal.RasterizeLayer( tmp_mask, [2], lyr, burn_values=[ 0 ] )   
    new_path_name = new_path +os.sep+ new_name+ '.tif'
    new_raster = gdal.GetDriverByName( 'GTiff' ).CreateCopy( new_path_name, tmp_mask, 0 )
    
    del new_raster, tmp_mask, lyr, shp_ply
    
    return new_path_name

# -----------------------------------------------------------------------------
def xyz2raster( x, y, z, new_name='new_raster', path='/vsimem/', extension='tif',
                prjcode=4326, nodata=9999, close=True ) :
    
    rx, ry, _ = utl.stepxy( x, y )
    lonmin = np.min( x ) - rx/2
    latmax = np.max( y ) + ry/2
    
    raster = array2raster( z, lonmin=lonmin, latmax=latmax, rx=rx, ry=ry, 
                           new_name=new_name, path=path, extension=extension, prjcode=prjcode,
                           nodata=nodata, close=close )
    
    return raster

# -----------------------------------------------------------------------------
def del_local_datasets() :
    
    path_names = []
    
    ld = locals()
    for k in ld :
        if type( ld[k] ) == gdal.Dataset :
            path_name = ld[k].GetDescription()
            ld[k] = None
            path_names.append( path_name )   
            
    gd = globals()
    for k in gd :
        if type( gd[k] ) == gdal.Dataset :
            path_name = gd[k].GetDescription()
            gd[k] = None
            path_names.append( path_name )               
            
    return  path_names       

# -----------------------------------------------------------------------------
def raster_mask( raster, shp_ply, path_name=None, plot=False, astype='raster' ) :
    
    if type(raster) == str:
        raster = gdal.Open(raster)
        
    shp_prjcode = shp.shp_prjcode( shp_ply ) 
    rst_prjcode = raster_prjcode( raster )
    lim = raster_lim( raster )
    lim_prj = raster_prjcode( raster )
    if shp_prjcode != rst_prjcode :
        lim_shp = utl.extend_lim( lim, 50, method='percentage')
        shp_ply = shp.translate( shp_ply, new_name=None, new_path='/vsimem/', 
                                 lim=lim_shp, lim_prjcode=lim_prj)          
   
    if type(shp_ply) == str:    
        shp_ply = ogr.Open( shp_ply )  
    
    lyr = shp_ply.GetLayer()   
    tmp_mask = gdal.GetDriverByName('MEM').CreateCopy('', raster, 0)
    tmp_mask.GetRasterBand(1).Fill(1)
    gdal.RasterizeLayer( tmp_mask, [1], lyr, burn_values=[ 0 ] ) 
    lyr = None
    
    if path_name != None :
        raster_mask = gdal.GetDriverByName( 'GTiff' ).CreateCopy( path_name, tmp_mask, 0 )
    else :
        raster_mask = tmp_mask
        
    if plot == True :
        pltr( raster_mask )
        
    if astype == 'raster' :
                
        return raster_mask
    
    if astype == 'bool' :
        
        return raster2array( raster_mask, nodata=0 )[2].astype(bool)

# -----------------------------------------------------------------------------
def mask_array( xx, yy, zz, shp_ply, prjcode=4326, plot=False, vmin=None, vmax=None ) :  

    zz_new = np.copy( zz ) 
    raster = array2raster( ( xx, yy, zz_new ), prjcode=prjcode, close=False )
    shp_mask_r = raster_mask( raster, shp_ply )
    shp_mask = raster2array( shp_mask_r, nodata=0 )[2].astype(bool)
    zz_new[shp_mask] = np.nan
    raster = None 
    shp_mask_r = None
    
    if plot is True :
        utl.plta( zz_new, vmin=vmin, vmax=vmax )
        
    return zz_new, shp_mask

        
    
