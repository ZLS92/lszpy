# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:58:52 2020

@author: lzampa
"""

import os
import pyproj as prj
import numpy as np
import scipy as sp
import platform
from scipy import signal
from matplotlib.widgets import LassoSelector
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import random
import matplotlib.pyplot as plt
import time 
import copy
import shapely 
from shapely.geometry import LineString, MultiPoint, MultiLineString 
import scipy.stats as sp_stat
from matplotlib.path import Path
from sklearn.decomposition import PCA
import datetime
from osgeo import gdal

# -----------------------------------------------------------------------------
# Constants

G = 6.6742*1e-11 # [m3/(kg *s^2)]
M = 5.97*1e24 # [kg]
a_wgs84 = 6378137 # [m]
c_wgs84 = 6356758 # [m]
R_wgs84 = ((a_wgs84**2)*c_wgs84)**(1/3) # [m]
J2_wgs84 = 1.081874*1e-3
w_wgs84 = 7.292115*1e-5 # [rad/sec]

# -----------------------------------------------------------------------------
def cm2in( cm ) :
    
    inches = cm * 1/2.54 
    
    return inches 

# -----------------------------------------------------------------------------
def tmi(t1=None, t2=None):
    """
    Print time interval from time t1 to t2
    --- If "t2 = None" , then, "t2 = now"
    """
    
    if ( t1 is None ) and ( t2 is None ) :
        return time.time()
    
    if t2 == None :
        print( time.time() - t1 )
    else:
        print( t2 - t1 )
        
# -----------------------------------------------------------------------------
def stat( array, decimals=None, show=True, out=None, multilines=False, sep='' ):
    """
    This function prints the statistics of a numpy array.

    Parameters:
    - array (numpy array): The input array for which the statistics are to be calculated.
    - decimals (int, optional): The number of decimal places to round to. 
        If None, no rounding is applied. Default is None.
    - show (bool, optional): If True, the function will print the statistics. 
        If False, the function will not print anything. Default is True.
    - out (str, optional): If 'str', the function will return the statistics as a string. 
        If None, the function will return the statistics as a tuple. Default is None.
    - multilines (bool, optional): If True, the function will print each statistic on a new line. 
        If False, all statistics will be printed on the same line. Default is False.
    - sep (str, optional): The separator between statistics when printed on the same line. 
        Default is an empty string.

    Returns:
    If out == 'str', returns a string containing the statistics.
    If out == None, returns a tuple containing the statistics (Min, Max, Mean, Std).
    """

    Min, Max, Mean, Std = ( np.nanmin(array), np.nanmax(array),
                            np.nanmean(array), np.nanstd(array) )
    

    if decimals != None:
        Min, Max, Mean, Std = ( np.round(Min, decimals), np.round(Max, decimals),
                                np.round(Mean, decimals), np.round(Std, decimals) )   
    
    len_ = np.max( ( len(str(Min)), len(str(Max)), len(str(Mean)), len(str(Std)) ) )      
    
    string = f"Min:{Min}{sep} Max:{Max}{sep} Mean:{Mean}{sep} Std:{Std}"
   
    if multilines == True :
        string = f"Min  = {Min: >{len_}} \nMax  = {Max: >{len_}} \nMean = {Mean: >{len_}} \nStd  = {Std: >{len_}}"
    
    if show == True :
        print( string ) 
        
    if out == str :
        return string
    
    else :     
        return Min, Max, Mean, Std    

# -----------------------------------------------------------------------------
def extend_lim( lim, d, method='percentage', sqr_area=False, plot=False, round2=None,
                print_lim=False ):
    """
    - lim = [xmin, xmax, ymin, ymax]
    - two methods aveilable:
        1) distance --> extend for a distance in the same units of lim
        2) percentage --> extend for a percentage of the total xy lengths of the area
    - sqr_area = True --> force the output boundaries to have length_X = length_Y
    """
    
    xd, yd = lim[1] - lim[0], lim[3] - lim[2]
    if sqr_area == True:
        if xd > yd: lim2 = [lim[0], lim[1], lim[2] - (xd - yd) / 2, lim[3] + (xd - yd) / 2]
        if yd > xd: lim2 = [lim[0] - (yd - xd) / 2, lim[1] + (yd - xd) / 2, lim[2], lim[3]]
        xd, yd = lim2[1] - lim2[0], lim2[3] - lim2[2]
    else: lim2 = lim

    if np.size(d) == 1: dx, dy = d, d
    if np.size(d) == 2: dx, dy = d[0], d[1]
    if method == 'percentage':
        xp, yp = xd * dx / 100, yd * dy / 100
        lim_new = [lim2[0] - xp, lim2[1] + xp, lim2[2] - yp, lim2[3] + yp]
    if method == 'distance': lim_new = [lim2[0] - dx, lim2[1] + dx, lim2[2] - dy, lim2[3] + dy]
    
    if round2 is not None :
        lim_new = [ round(l, round2) for l in lim_new ]

    if plot == True:
        xplot1 = [lim[0], lim[0], lim[1], lim[1], lim[0]]
        yplot1 = [lim[2], lim[3], lim[3], lim[2], lim[2]]
        xplot2 = [lim_new[0], lim_new[0], lim_new[1], lim_new[1], lim_new[0]]
        yplot2 = [lim_new[2], lim_new[3], lim_new[3], lim_new[2], lim_new[2]]
        plt.plot(xplot1, yplot1, c='r')
        plt.plot(xplot2, yplot2, c='b')

    if print_lim == True :
        print( lim_new )
        
    return lim_new

# -----------------------------------------------------------------------------
def prj_( prjcode ):
    """
    Convert epsg or proj4 string to proj4 code object
    """
    if type( prjcode )==int:
        prj4_obj = prj.CRS( 'epsg:'+str( prjcode ) )
        
    if type( prjcode )==str:
        prj4_obj = prj.CRS( prjcode )
        
    return prj4_obj

# -----------------------------------------------------------------------------
def prj_units( prjcode ):
    """
    Return EPGS or Proj4 code units ( i.e., meters or degree )
    """
    
    prj4_obj = prj_( prjcode )
    units = prj4_obj.axis_info[0].unit_name
        
    return units

# -----------------------------------------------------------------------------
def prjxy( prjcode_in, prjcode_out, x, y, z=None ):
    """
    Transform coordinates from a reference system to another
    """
    
    x = np.copy( x )
    y = np.copy( y )
    
    if prjcode_in == prjcode_out :
        prj_coord = x, y
        
    else :
        prj_in = prj_( prjcode_in )
        prj_out = prj_( prjcode_out )
        trans = prj.Transformer.from_crs( prj_in, prj_out, always_xy=True )
        if z != None :
            prj_coord = trans.transform( x, y, z )
        if z == None :
            prj_coord = trans.transform( x, y )

    return prj_coord

# -----------------------------------------------------------------------------
def prj2epsg( prjcode ) :
    
    crs = prj.CRS( prjcode )
    epsg = crs.to_epsg()
    
    return epsg

# -----------------------------------------------------------------------------
def lim_sort( lim, p=False ):

    lim_out = [lim[0], lim[2], lim[1], lim[3]]

    if p == True :
        print( lim_out )

    return lim_out

# -----------------------------------------------------------------------------
def lim2points( lim ) :

    x = np.array( ( lim[0], lim[1], lim[1], lim[0] ) )
    y = np.array( ( lim[3], lim[3], lim[2], lim[2] ) )

    return x, y

# -----------------------------------------------------------------------------
def prj_lim(lim, prjcode_in, prjcode_out, sort='xlyl'):
    """
    Transform limits of an area (i.e. [LonMin, LonMax, LatMin, LatMax]) from a
    reference system to another
    """
    if sort=='xlyl':
        x,y = prjxy(prjcode_in, prjcode_out, (lim[0],lim[1]), (lim[2],lim[3]))
    if sort=='xyl':
        x,y = prjxy(prjcode_in, prjcode_out, (lim[0],lim[2]), (lim[1],lim[3]))

    limf = [x[0],x[1],y[0],y[1]]

    return limf

# -----------------------------------------------------------------------------
def prj_centre( xy_centre, prj_type, ellps='WGS84' ):
    """
    Creates a PROJ string code, with a reference system centered (i.e tangent)
    to a defined point on Earth surface: xy_centre,
    i.e. [Lon_centre, Lat_centre] (Lat&Lon in geographic coordinates)
    """
    x_centre, y_centre = xy_centre
    if type(prj_type)==str:
       prj_cent_str = f'''+proj={prj_type} +lat_0={x_centre} +lon_0={y_centre} +ellps={ellps}'''
    if type(prj_type)==int:
       prj_cent_str = f'''+epsg={prj_type} +lat_0={x_centre} +lon_0={y_centre} +ellps={ellps}'''

    return prj_cent_str

# -----------------------------------------------------------------------------
def xy_in_lim( x, y, lim, extend=0, method='percentage', plot=False, s=1, plot_lim=None ):
    """
    Return all xy points within the given limits [xmin, xmax, ymin, ymax]
    """
    
    if extend != 0 :
        lim = extend_lim( lim, d=extend, method=method )
    
    idx = (x >= lim[0]) & (x <= lim[1]) & (y >= lim[2]) & (y <= lim[3])
    xl = x[ idx ]
    yl = y[ idx ]
    
    if plot is True :
        plt.scatter( x[~idx], y[~idx], c='b', s=s )
        plt.scatter( xl, yl, c='r', s=s )
        if plot_lim is not None : 
            lim_ext = extend_lim( lim, plot_lim )
            plt.xlim( lim_ext[0], lim_ext[1] )
            plt.ylim( lim_ext[2], lim_ext[3] )
        
    return xl, yl, idx

# -----------------------------------------------------------------------------
def find_by_dist( xy1, xy2, dist, prj_dist=None, prj_in=4326, plot=False ):
    
    """
    Find all points of a set (xy1) within a distance (dist)
    from another set of points (xy2)
    """

    set1 = np.column_stack(xy1)
    set2 = np.column_stack(xy2)
    if prj_dist is not None:
       set1[:,0], set1[:,1] = prjxy(prj_in, prj_dist, set1[:,0], set1[:,1])
       set2[:,0], set2[:,1] = prjxy(prj_in, prj_dist, set2[:,0], set2[:,1])

    f_dist = lambda i: np.any(np.sqrt((i[0]-set2[:,0])**2 + (i[1]-set2[:,1])**2)<=dist)
    set1_idx = np.array((list(map(f_dist, set1))))
    set1_new = set1[set1_idx]

    if plot==True:
        plt.scatter(set1[:,0],set1[:,1], c='g', label='set1')
        plt.scatter(set2[:,0],set2[:,1], c='b', label='set2')
        plt.scatter(set1_new[:,0],set1_new[:,1], marker='+', c='r',
                    label=f'''set1 points, {dist} distant from set2 points''')
        plt.legend()

    if prj_dist is not None:
        set1_new[:,0], set1_new[:,1] = prjxy(prj_dist, prj_in, set1_new[:,0], set1_new[:,1])

    return set1_idx, set1_new

# -----------------------------------------------------------------------------
def min_dist( x, y, prjcode_in=4326, prjcode_out=4326, 
              data_type='scattered_points', factor=1000 ) :
    """
    Find the minimum average distance among a set of scattered points.
    The result is a dictionary contatnjing also other statistics.
    """
    
    if prjcode_in != prjcode_out :
        x, y = prjxy( prjcode_in, prjcode_out, x, y )

    if data_type == 'grids' :
        dx, dy, mdxy = stepxy( x, y )
        d = {'dx':dx, 'dy':dy, 'mean':mdxy }        
        
    if  data_type == 'scattered_points' :   
        points = np.column_stack( ( x.ravel(), y.ravel() ) )
        N = points.shape[0]
        md = np.zeros( N )

        # for i in range( N ):
        #     apoints = np.delete( points, i, 0 )
        #     d = np.sqrt( ( points[(i, 0)] - apoints[:, 0] ) ** 2 + \
        #                  ( points[(i, 1)] - apoints[:, 1] ) ** 2 )
        #     md[ i ] = np.min( d )

        # --------------------------------------
        kdtree = sp.spatial.cKDTree( points )
        distances, _ = kdtree.query( points, k=2 )
        md = np.min( distances[:, 1:], axis=1 )
        # --------------------------------------

        meand = np.mean(md)
        mind = np.min(md)
        maxd = np.max(md)
        stdd = np.std(md)
        
        # moded = sp_stat.mode(md, nan_policy='omit')[0][0]
        d = { 'mean':meand,'val':md, 'min':mind, 'max':maxd, 'std':stdd }
    
    return d

# -----------------------------------------------------------------------------
def xy2XY( x, y, step=None, lim=None, dist='mean', method='distance', extend=False,
           plot=False, grid=False, rotate=True, treshold=1e-5, fix_point='first',
           s=None, return_step=False ) :

    if grid is True :
        
        if rotate == True :
            angle, x, y, fix_point = grid_rot_angle( x, y, treshold=treshold, fix_point=fix_point ) 
            
    if type( step ) in ( int, float ) :
        step = [ step, step ]
        
        if lim is None :
            lim = xy2lim( x, y, extend=extend, method=method )
            
        xm = np.arange( lim[0], lim[1]+step[0], step[0] )  
        ym = np.arange( lim[3], lim[2]-step[1], -step[1] ) 

    else :
        xm = np.unique( x )
        ym = np.unique( y )
    
    X, Y = np.meshgrid( xm, ym )
    
    if grid == True :
        if rotate == True :
            X, Y, _ = rotate_xy( X, Y, radians=-angle, fix_point=fix_point )
    
    if plot is True :
        plt.scatter( X.ravel(), Y.ravel(), c='r', s=s )
        
    if return_step == True :
        
        return X, Y, step
    
    else :
        
        return X, Y
        
# -----------------------------------------------------------------------------
def xyz2xy( xyz, xy, method='nearest', algebra='diff', rescale=False, fillnan=True,
            lim=None, plot=False ) :

    x, y, z = xyz
    
    nan = np.isnan( z )
    
    x1 = x[ ~nan ]
    y1 = y[ ~nan ]
    z1 = z[ ~nan ]
    
    if len( xy ) == 2 :
        x2, y2 = xy
    if len( xy ) > 2 : 
        x2, y2, z2 = xy
    
    if np.size( x1 ) < 4 :
        method = 'nearest'
    
    try:    
        zn = sp.interpolate.griddata( ( x1, y1 ), z1, ( x2, y2 ), method=method, 
                                      rescale=rescale )
    except : 
        zn = sp.interpolate.griddata( ( x1, y1 ), z1, ( x2, y2 ), method='nearest', 
                                      rescale=rescale )        
            
    if fillnan is True :
        
        if ( np.size( zn ) == 1 ) and ( np.any( np.isnan( zn ) ) ) :
            zn = sp.interpolate.griddata( ( x1, y1 ), z1, ( x2, y2 ), method='nearest' ) 
            
        if ( np.size( zn ) > 1 ) and ( np.any( np.isnan( zn ) ) ) :   
            idx = np.isnan( zn )
            zn[ idx ] = sp.interpolate.griddata( ( x1, y1 ), z1, ( x2[idx], y2[idx] ), 
                                                  method='nearest' )
            
    if len( xy ) > 2 : 
        if algebra == 'diff':
            za = z2 - zn
    else:
        za = None         

    if plot is True :
        plt.figure()
        sts = stat( zn )
        plt.scatter( xy[0], xy[1], c=zn, vmin=sts[2]-sts[3]*2, vmax=sts[2]+sts[3]*2, cmap='rainbow' )
        plt.colorbar()

    return zn, za

# -----------------------------------------------------------------------------
def find_by_att( dat1, dat2, col_xy, delta_xy, col_att, delta_att, condition=True,
                 prj_in=4326, prj_dist=None, unique=True, plot=False ) :

    dat1c = np.copy(dat1)
    dat2c = np.copy(dat2)
    
    if prj_dist is not None:
        dat1c[:,col_xy[0]], dat1c[:,col_xy[1]] = prjxy(prj_in, prj_dist, dat1[:,col_xy[0]], dat1[:,col_xy[1]])
        dat2c[:,col_xy[0]], dat2c[:,col_xy[1]] = prjxy(prj_in, prj_dist, dat2[:,col_xy[0]], dat2[:,col_xy[1]])
        
    mask1 = np.full(dat1c.shape[0], False)
    idx21 = []
    uniq = False
    col = col_xy + col_att
    delta = delta_xy + delta_att
    
    for n, i in enumerate(dat2c):
        mask0 = np.full(dat1c.shape[0], True)
        
        for c, d in zip(col, delta):
            msk = (dat1c[:,c] >= i[c] - d ) & ( dat1c[:,c] <= i[c] + d )
            mask0 = mask0 & msk
            
        if np.any(mask0):
            idx21.append([n, mask0])
            if ( np.sum(idx21[-1][1]) > 1 ) and ( unique is True ):
               uniq= uniq | True
               dist = np.sqrt( ( dat1c[mask0,col_xy[0]]-i[1] )**2  + 
                               ( dat1c[mask0,col_xy[1]]-i[0] )**2  )
               idx210N = np.nonzero( mask0*1 )[0]
               min_dist = np.nanmin( dist )
               unid = idx210N[ dist == min_dist ]                   
               idx21[-1][1] = unid[0] 
            if ( np.sum(idx21[-1][1] ) == 1 ): 
                idx21[-1][1] = np.where(mask0)[0].item()
            if ( np.sum(idx21[-1][1]) > 1 ) and ( unique is False ):
                uniq = uniq | True
                idx21[-1][1] = np.where(mask0).tolist()
        mask1 =  mask1 | mask0


    if unique is True:
        idx21 = np.asarray(idx21)

    if condition==False: mask1 = ~mask1
    dat1_msk = dat1[mask1]

    print( 'All data: ', len(dat1) )
    print( 'True: ', len(dat1_msk) )
    print( 'False: ', len(dat1)-len(dat1_msk) )

    if plot==True:
        plt.scatter(dat1[:,col_xy[0]], dat1[:,col_xy[1]], c='g', label='set1')
        plt.scatter(dat2[:,col_xy[0]], dat2[:,col_xy[1]], c='b', label='set2')
        plt.scatter(dat1_msk[:,col_xy[0]], dat1_msk[:,col_xy[1]], marker='+', c='r',
                    label='set1 points, defined by condition')
        plt.legend()

    return mask1, dat1_msk, idx21

# -----------------------------------------------------------------------------
def ell_radius( lat, radians=False ) :
    """
    This function calculates the radius of the Earth at a given latitude 
    based on the WGS84 ellipsoid model.

    Parameters:
    - lat (float or np.array): Latitude at which the radius is to be calculated. 
        It can be a single value or a numpy array of values. 
        The latitude should be in degrees by default. 
        If the latitude is in radians, the 'radians' parameter should be set to True.

    - radians (bool, optional): A flag to indicate if the input latitude is in radians. 
        Default is False, which means the function expects the latitude in degrees. 
        If this is set to True, the function will treat the input latitude as radians.

    The WGS84 ellipsoid model is defined by the semi-major axis (a_wgs84), 
    and the semi-minor axis (c_wgs84). 
    The function calculates the radius (R) at the given latitude, 
    using the formula for the radius of curvature in the plane of the meridian.

    Returns:
    - R (float or np.array): The radius of the Earth at the given latitude(s) 
        based on the WGS84 ellipsoid model. 
        The radius is returned in the same units as the semi-major and semi-minor axes.
    """

    if radians is False :
        lat = np.copy( np.deg2rad( lat ) )
        
    num = ( ( a_wgs84**2 * np.cos( lat ) )**2 ) + ( ( c_wgs84**2 * np.sin( lat ) )**2 )   
    den = ( ( a_wgs84 * np.cos( lat ) )**2 ) + ( ( c_wgs84 * np.sin( lat ) )**2 )  
    
    R = np.sqrt( num / den )
    
    return R

# -----------------------------------------------------------------------------
def local_sph_raduis(lat):
    """
    Radius of the local sphere (based on latitude)

    Ref:
    - http://www2.ing.unipi.it/~a009220/lezioni/LI_ING_EDILE/AA1011/MATERIALE_DIDATTICO/APPUNTI/Geodesia.pdf

    """

    lat_r = np.radians( lat )
    
    e2 = ( a_wgs84**2 - c_wgs84**2 ) / a_wgs84**2
    
    N = a_wgs84 / np.sqrt( 1 - e2 * np.sin( lat_r )**2 )
    
    rho = a_wgs84 * ( 1 - e2 ) / np.sqrt( ( 1 - e2 * np.sin( lat_r )**2 )**3 )
    
    R = np.sqrt( rho * N )

    return R

# -----------------------------------------------------------------------------
def m2deg(md, n_digits=9, R=R_wgs84, lat=None):
    """
    Convert metric distances to decimal degrees (spherical approximation)

    lat = if not None, it calculate the distance using the Local Sphere Approximation,
            tangent to the wgs84 ellipsoid at the given latitude
    """

    if lat is None:   
        radi = R
    else:   
        radi = local_sph_raduis(lat)

    dd = np.round( md*360 / ( 2 * np.pi * radi ), n_digits)

    return dd

# -----------------------------------------------------------------------------
def deg2m(dd, lat=None, n_digits=3, R=R_wgs84):
    """
    Convert deg distances to meters (spherical approximation)

    lat = if not None, it calculate the distance using the Local Sphere Approximation,
            tangent to the wgs84 ellipsoid at the given latitude
    """

    if lat is None:   
        radi = R
    else:   
        radi = local_sph_raduis(lat)

    md = np.round( dd * 2 * np.pi * radi / 360, n_digits )

    return md

# -----------------------------------------------------------------------------
def pad_array( array, padw=25, mode='edge', alpha=None, constant_values=np.nan,
               plot=False, vmin=None, vmax=None, method='gdal', iter=1, 
               ptype='percentage', sqr_area=False ):
    
    ny, nx = array.shape
    
    if type( padw ) in ( int , float ) :
        padw = ( padw, padw ) 
    
    if ptype == 'percentage' :
        padw = [ int( ny * padw[0] / 100 ), int( nx * padw[1] / 100 ) ]
        
    if sqr_area == True:
        if ny > nx : padw[1] = padw[1] + ny - nx
        if ny < nx : padw[0] = padw[0] + nx - ny
            
    
    if mode in ('surface', 'gdal'): 
        pad_array = np.pad( array, pad_width=( ( padw[0], padw[0] ), ( padw[1], padw[1] ) ), 
                            mode='constant', constant_values=np.nan )
        pad_array = fillnan( pad_array, method=mode, iter=iter )      
    elif mode == 'constant' : 
        pad_array = np.pad( array, pad_width=((padw[0], padw[0]), (padw[1], padw[1])), 
                            mode=mode, constant_values=constant_values )        
    else: 
        pad_array = np.pad( array, pad_width=((padw[0], padw[0]), (padw[1], padw[1])), 
                            mode=mode )
    
    pnx, pny = pad_array.shape
    y0, y1, x0, x1 = (padw[0], pnx - padw[0], padw[1], pny - padw[1])
    original_shape_indx = (y0, y1, x0, x1)
        
    if alpha is not None:
        pad_array = taper( pad_array, alpha=alpha )[0]
        
    if plot == True:
        plta( array, sbplt=[1,2,1], vmin=vmin, vmax=vmax )
        plta( pad_array, sbplt=[1,2,2], vmin=vmin, vmax=vmax )
        
    return pad_array, original_shape_indx

# -----------------------------------------------------------------------------
def pad_xx_yy( pad_arr, xx, yy, plot=False ):
    
    dx = abs( np.mean(np.diff( xx, axis=1 ) ) )
    dy = abs( np.mean(np.diff( yy, axis=0 ) ) )
    
    a, c = pad_arr[0], pad_arr[1]
    xmin1, xmax1 = np.min(xx), np.max(xx)
    ymin1, ymax1 = np.min(yy), np.max(yy)
    xmin2, xmax2 = xmin1 - dx * c[2], xmax1 + dx * (a.shape[1] - c[3])
    ymax2, ymin2 = ymax1 + dy * c[0], ymin1 - dy * (a.shape[0] - c[1])
    xv = np.linspace(xmin2, xmax2, a.shape[1])
    yv = np.linspace(ymax2, ymin2, a.shape[0])
    xxn, yyn = np.meshgrid( xv, yv )
    
    if plot == True:
        plt.figure()
        plt.scatter((xxn.flatten()), (yyn.flatten()), c='b')
        plt.scatter((xx.flatten()), (yy.flatten()), c='r')
        
    return xxn, yyn
    
# -----------------------------------------------------------------------------
def taper( array, alpha=0.5, plot=False, vmin=None, vmax=None ):
    
    nx, ny = array.shape[1], array.shape[0]
    tape_x = signal.tukey(nx, alpha=alpha)
    tape_y = signal.tukey(ny, alpha=alpha)
    
    t_xx, t_yy = np.meshgrid( tape_x, tape_y )
    taper_filt = t_xx * t_yy
    tarray = taper_filt * ( array - np.nanmean( array ) )
    tarray = tarray + np.nanmean( array )
          
    
    if plot == True:
        
        plta( array, sbplt=[1,3,1], vmin=vmin, vmax=vmax )
        plta( taper_filt, sbplt=[1,3,2], vmin=vmin, vmax=vmax )
        plta( tarray, sbplt=[1,3,3], vmin=vmin, vmax=vmax )
    
    return tarray, taper_filt        

# -----------------------------------------------------------------------------
def crop_pad(array, original_shape_indx, plot=False, vmin=None, vmax=None):
    
    array_crop = array[original_shape_indx[0]:original_shape_indx[1],
    original_shape_indx[2]:original_shape_indx[3]]
    
    if plot == True:
        
        plta( array, sbplt=[1,2,1], vmin=vmin, vmax=vmax )
        plta( array_crop, sbplt=[1,2,2], vmin=vmin, vmax=vmax )
        
    return array_crop

# -----------------------------------------------------------------------------
def xy2lim( x, y, prjcode_in=4326, prjcode_out=4326, extend=False, d=0,
            method='distance', sqr_area='False', plot=False ):

    if prjcode_in != prjcode_out:
        x,y = prjxy(prjcode_in, prjcode_out, x, y)

    lim = [ np.min(x), np.max(x), np.min(y), np.max(y) ]

    if extend is True:
        lim = extend_lim(lim, d, method, sqr_area )

    if plot is True:
        xplot = [lim[0], lim[0], lim[1], lim[1], lim[0]]
        yplot = [lim[2], lim[3], lim[3], lim[2], lim[2]]
        plt.scatter(x, y, marker='+')
        plt.plot(xplot, yplot, c='r')

    return lim

# -----------------------------------------------------------------------------
def absolute_file_paths( directory ):

    path = os.path.abspath(directory)
    
    return [entry.path for entry in os.scandir(path) if entry.is_file()]

# -----------------------------------------------------------------------------
def array2step( arrays, axis=0 ) :

    if type( arrays ) != list :
        arrays = [arrays]
        
    if type( axis ) != list :
        axis = [axis]
        
    steps = []
    
    for i, a in zip( arrays, axis ):
        
        if len( i.shape ) == 1 :
            data = i.ravel()
            step = np.nanmean( np.diff( np.unique( data ) ) )
            
        if len( i.shape ) == 2 :    
            
            if a == 0 :
                data = i[ :, 0 ]
            if a == 1 :
                data = i[ 0, : ]            
            step = np.nanmean( np.diff( np.unique( data ) ) ) 
                
        if len( i.shape ) == 3 :    
            
            if a == 0 :
                data = i[ :, 0, 0 ]
            if a == 1 :
                data = i[ 0, :, 0 ]  
            if a == 2 :
                data = i[ 0, 0, : ] 
                                
            step = np.nanmean( np.diff( np.unique( data ) ) ) 
                
        steps.append( step )

    return steps

# -----------------------------------------------------------------------------
def plta( array, vmin=None, vmax=None, tit=None, lim=None, stat=6, sbplt=[],
          cmap='rainbow', axis=False, new_fig=True, contours=[], adjst_lim=True,
          flipud=False, hillshade=False, ve=2, aspect='auto', blend_mode='overlay',
          mask=None, points=None, pc='k', ps=1, label=None, label_size='large',
          xlabel=None, ylabel=None, x_ax=True, y_ax=True, letter=None, 
          xlett=0, ylett=0, colorbar=True, print_stat=True, alpha=1, lines=None,
          lc='k', lett_size='large', lett_colour='k', cc=None, out_lim=None,
          cl=True, resemp_fac=None ) :
    
    """
    Plot 2D array and print statistics
    """
    
    array = np.copy( array )
    
    if mask is not None :
        array[ mask ] = np.nan
        
    Min, Max, Mean, Std = np.nanmin(array), np.nanmax(array), \
                          np.nanmean(array), np.nanstd(array) 
    
    cmpa = plt.cm.get_cmap( cmap )
    
    if stat != None:
        if type(tit) == int : stat = tit
        Min, Max, Mean, Std = np.round(Min, stat), np.round(Max, stat), \
                              np.round(Mean, stat), np.round(Std, stat) 
                              
        if print_stat == True :
        
            print(f"Min:{Min} Max:{Max} Mean:{Mean} Std:{Std}")

    if sbplt != [] :
        r, c, n = sbplt
    else :
        r, c, n = 1, 1, 1
        
    if ( new_fig is True ) and ( n == 1 ) :
        plt.figure()  
        
    if sbplt != []:
        plt.subplot( r, c, n )
        
    if vmin == None:
        vmin = Mean - 2 * Std
    if vmax == None:
        vmax = Mean + 2 * Std
    
    if lim != None :
        lim = copy.copy( lim )
        if len( lim ) == 2 :
            lim = xy2lim( x=lim[0], y=lim[1] )
        dx = ( lim[1] - lim[0] ) / array.shape[0] 
        dy = ( lim[3] - lim[2] ) / array.shape[1] 
        
        if adjst_lim == True :
            lim[0], lim[2] = lim[0] - dx/2, lim[2] - dy/2 
            lim[1], lim[3] = lim[1] + dx/2, lim[3] + dy/2
    else :
        dx, dy = 1, 1
        
    if flipud == True :
        array = np.flipud( array )

    if resemp_fac is not None :
        array = resampling( array, resemp_fac, spl_order=1, dimention='2D', mode='nearest' )
    
    ax = plt.imshow( array, vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap, extent=lim,
                     alpha=alpha )
    plt.xlabel( xlabel, fontsize=label_size )
    plt.ylabel( ylabel, fontsize=label_size )
    
    if letter is not None :
        plt.annotate(letter, xy=(xlett,ylett), xycoords='axes fraction', size=lett_size,
                     c=lett_colour ) 
    
    ax = plt.gca()
    if colorbar is True :  
        cax = make_axes_locatable(ax).append_axes( 'right', size='5%', pad=0.05 ) 
        clb = plt.colorbar( cax=cax )
        
        if label is not None :
            clb.set_label( label, size=label_size )
                         
    if hillshade == True :
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade( array, cmap=cmpa, blend_mode=blend_mode, vert_exag=ve,
                        vmin=vmin, vmax=vmax, dx=1, dy=1 )
        alpha2 = np.isnan( array )
        rgb[ alpha2 ] = np.nan
        ax.imshow( rgb, extent=lim, alpha=alpha, aspect=aspect  )
    
    if points is not None :
        ax.scatter( points[0], points[1], c=pc, s=ps )

    if lines is not None :
        for line in lines :  
            ax.plot( line[0], line[1], c=lc)
        ax.set_xlim( lim[0], lim[1] )
        ax.set_ylim( lim[2], lim[3] )
    
    if contours != [] :
        cont = ax.contour( np.flipud(array), levels=contours, extent=lim, colors=cc,
                    linestyles='solid')  
        
        if cl is True :
            plt.clabel( cont )
        
    if axis is False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)   

    if x_ax is False:
        ax.axes.xaxis.set_visible(False)
    if y_ax is False:
        ax.axes.yaxis.set_visible(False)        
        
    if type(tit) == str:
        ax.set_title(tit)
    if type(tit) == int:
        Min, Max, Mean, Std = np.round(Min, tit), np.round(Max, tit), \
                              np.round(Mean, tit), np.round(Std, tit)
        tit = f"[ Min={Min}  Max={Max}  Mean={Mean}  Std={Std} ]"
        ax.set_title(tit)
        
    if ( out_lim is not None ) and ( out_lim != [] ) :
        ax.set_xlim( [ out_lim[0], out_lim[1] ] )
        ax.set_ylim( [ out_lim[2], out_lim[3] ] )
        
    return ax
        
# -----------------------------------------------------------------------------
def map_profile( xy_start, xy_end, X, Y, Z=[], step=None, 
                 yaxis_sep=50, method='nearest',
                 colors=None,
                 legend_labels=None,
                 legend_label_size=None,
                 markers=None,
                 linestyle=None,
                 y_labels=None,
                 legend_ncol=1,
                 legend_loc='best',
                 x_label=None,
                 ax_label_size=None,
                 font='Tahoma',
                 y_ax_lim=None,
                 x_ax_lim=None,
                 plot=True,
                 subplot=(1, 1, 1),
                 text=None, text_size=14,
                 text_boxstyle='round',
                 text_facecolor='wheat',
                 text_alpha=0.5,
                 text_position=(0.95, 0.05),
                 xaxis_visible=True,
                 prjcode_in=4326,
                 smooth=1,
                 prjcode_out=None,
                 m2km=False,
                 deg2km=False,
                 prjcode_limits=None,
                 x_axis = True,
                 plot_map=False,
                 vmin=None,
                 vmax=None,
                 y_axis_type='multiple',
                 new_fig=True):

    from matplotlib import rcParams

    rcParams['font.family'] = font
    if prjcode_out is None:
        prjcode_out = prjcode_in

    if prjcode_limits is None:
        prjcode_limits = prjcode_in

    if prjcode_out != prjcode_in:
        X, Y = prjxy(prjcode_in, prjcode_out, X, Y)

    if prjcode_limits != prjcode_out:
        xy_start = prjxy(prjcode_limits, prjcode_out, xy_start[0], xy_start[1])
        xy_end = prjxy(prjcode_limits, prjcode_out, xy_end[0], xy_end[1])

    print( 'line step:', step )
    if step is None:
        step = stepxy( X, Y )[2]
        print(step)
    x0, x1 = xy_start[0], xy_end[0]
    y0, y1 = xy_start[1], xy_end[1]
    theta = np.arctan((y1 - y0) / ((x1 - x0)+1e-24) )
    length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    lp = np.arange(0, length, step/smooth)
    xp = x0 + lp * np.cos(theta)
    yp = y0 + lp * np.sin(theta)
    
    if m2km is True :
        lp /= 1000
        
    if deg2km is True :
        lp = deg2m(lp)/1000  
        
    profiles = []
    for i, arr in enumerate(Z):
#        arr[np.isnan(arr)]=0
        lim = extend_lim( [ np.min((x0,x1)), np.max((x0,x1)), 
                            np.min((y0,y1)), np.max((y0,y1)) ], 2, sqr_area=True )
        xl, yl, idx = xy_in_lim( X, Y, lim )
        al = arr[idx]
        profiles.append(sp.interpolate.griddata((xl.ravel(), yl.ravel()), (al.ravel()),
                                                 (xp, yp), method=method, fill_value=np.nan))
        
    if plot is True:
        if new_fig is True :
            plt.figure()
        if colors is None:
            colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(profiles))]
        if legend_labels is None:
            legend_labels = ['function_' + str(i) for i in range(len(profiles))]
        if markers == 'random':
            markers_rand = itertools.cycle( ( ',', '+', '.', '*', '<', '>', 'h', 'p', 's',
                                              '.', 'H', 'D' ) )
        if linestyle is None :
            linestyle = ['solid' for i in range( len( profiles) ) ]
            
        lns = []
        ax_orig = plt.subplot(subplot[0], subplot[1], subplot[2])
        
        for i, (arr, color, lnsty) in enumerate(zip(profiles, colors, linestyle)):         
            if ( i == 0 ) or ( y_axis_type == 'unique') :
                ax = ax_orig
            else:
                ax = ax_orig.twinx()
                ax.spines['right'].set_position(('outward', yaxis_sep * (i - 1)))
            if markers == 'random' :
                ln = ax.plot(lp, arr, color=color, marker=(next(markers_rand)), linestyle=lnsty)
            else:
                ln = ax.plot(lp, arr, color=color, linestyle=lnsty )
            if y_axis_type == 'unique' :    
                ax.tick_params(axis='y', colors='k')
            else :
                ax.tick_params(axis='y', colors=color )
            if y_labels is not None:
                plt.ylabel(y_labels, fontsize=ax_label_size)
            if x_label is not None:
                plt.xlabel(x_label, fontsize=ax_label_size)
            lns += ln
            if y_ax_lim is not None :
                ax.set_ylim( y_ax_lim[i] )      
            if x_ax_lim is not None :
                ax.set_xlim( x_ax_lim[i] )                          

        if text is not None:
            props = dict(boxstyle=text_boxstyle, facecolor=text_facecolor, alpha=text_alpha)
            ax_orig.annotate(text, xy=text_position, xycoords='axes fraction',
              fontsize=text_size,
              bbox=props)
            plt.tight_layout()

        if legend_labels is not None:
            ax_orig.legend(lns, legend_labels, fontsize=legend_label_size, 
                                 ncol=legend_ncol, loc=legend_loc, framealpha=1)
            ax_orig.axes.get_xaxis().set_visible(xaxis_visible)
    
        if x_axis is False :
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            
    if plot_map == True:
        plt.figure()
        if vmin == None:
            vmin = np.nanmean(Z[0]) - 2 * np.nanstd(Z[0])
        if vmax == None:
            vmax = np.nanmean(Z[0]) + 2 * np.nanstd(Z[0])
        plt.imshow(Z[0], vmin=vmin, vmax=vmax, cmap='rainbow',
                   extent=(X.min(), X.max(), Y.min(), Y.max()))
        plt.plot(xp, yp, color='k')            
    
    pr_list = []
    for i in profiles:
        pr_list.append(np.column_stack((xp, yp, lp, i)))

#    plt.tight_layout()

    return pr_list

# -----------------------------------------------------------------------------
def XYZ_crop2lim( X, Y, Z, lim ) :

    if len( Z.shape ) == 2:

        xi = np.where( ( X[0, :] >= lim[0]) & (X[0, :] <= lim[1] ) )
        yi = np.where( ( Y[:, 0] >= lim[2]) & (Y[:, 0] <= lim[3] ) )

        Xc = X[ np.min(yi):np.max(yi), np.min(xi):np.max(xi) ]
        Yc = Y[ np.min(yi):np.max(yi), np.min(xi):np.max(xi) ]
        Zc = Z[ np.min(yi):np.max(yi), np.min(xi):np.max(xi) ]
        

        return [ Xc, Yc, Zc ]

    if len( Z.shape ) == 1:

        xi = np.where( ( X >= lim[0] ) & ( X <= lim[1] ) )
        yi = np.where( ( Y >= lim[2] ) & ( Y <= lim[3] ) )

        xc = X[ ( xi, yi ) ]
        yc = Y[ ( xi, yi ) ]
        zc = Z[ ( xi, yi ) ]

        return [ xc, yc, zc ]

# -----------------------------------------------------------------------------
def isiterable(p_object):
    """
    Check if an object is iterable
    """

    try:
        it = iter(p_object)
    except TypeError:
        return False
    return True

# -----------------------------------------------------------------------------
def stepxy( xarray, yarray ) :
    
    " Distance between 2Darray grid data points "
    
    cx = int( xarray.shape[1]/2 )
    cy = int( yarray.shape[0]/2 )
    
    arx = xarray[ cy:cy+2, cx:cx+2 ]
    ary = yarray[ cy:cy+2, cx:cx+2 ]
    
    dx = np.sqrt( ( arx[0,1] - arx[0,0] )**2 + ( ary[0,1] - ary[0,0] )**2 )
    dy = np.sqrt( ( arx[0,0] - arx[1,0] )**2 + ( ary[0,0] - ary[1,0] )**2 )
    
    dmxy = np.mean( ( dx, dy ) )
        
    return dx, dy, dmxy

# -----------------------------------------------------------------------------
def gmt_surf( x, y, z, gstep=None, lim=None, pause=False, filt='', blkm='', grid_shape=None,
              convergence_limit='', max_iterations=100, max_radius='', Q=False,
              search_radius='0.0s', tension_factor=0.35, relaxation_factor=1.4,
              remove_files=True, plot=False, vmin=None, vmax=None ):
             
    nan = np.isnan(z)
    x, y, z = x[(~nan)], y[(~nan)], z[(~nan)]

    if lim is None:
        lim = (np.min(x),np.max(x),np.min(y),np.max(y))
    x0, x1, y0, y1 = (round(lim[0], 5), round(lim[1], 5), round(lim[2], 5), round(lim[3], 5))

    if ( gstep is None ) and ( grid_shape is None ) :
        gstep = np.round( min_dist(x, y)['mean'], 4 )
        gs = '-I' + str(gstep)
        blkm = gstep
        print( 'gstep : ', gstep)

    if gstep is not None: 
        gs = '-I' + str(gstep)

    if blkm != '' :
        in_file = 'xyz_temp_blkm'
    else :
        in_file = 'xyz_temp'

    Rlim = f"-R{x0}/{x1}/{y0}/{y1}"
    xyz_temp = np.column_stack((x, y, z))
    np.savetxt('xyz_temp', xyz_temp, fmt='%f')

    if pause == True:
        pause = 'pause'
    if pause == False:
        pause = ''
    if blkm != '':
        blkm = f"gmt blockmean xyz_temp -I{blkm} -R{x0}/{x1}/{y0}/{y1} > xyz_temp_blkm"
    if filt != '':
        filt = f"gmt grdfilter surf_temp.grd -Fg{filt} -Gsurf_temp.grd -Dp"
    if grid_shape != None :
        gs = f"-I{grid_shape[1]}+n/{grid_shape[0]}+n"
        
    if max_radius != '' :
        if type( max_radius ) in (int, float) :
            max_radius = f"-M{max_radius}c"
        if type( max_radius ) == str :
            max_radius = f"-M{max_radius}"            

    surface = f"gmt surface {in_file} {gs} {Rlim} " \
              f"-Gsurf_temp.grd -T{tension_factor} -N{max_iterations} " \
              f"-C{convergence_limit} {max_radius} "
              
    if platform.system() == 'Linux' :
        ext = '.sh'
        pre= './'
    else :
        ext = '.bat'
        pre = ''
    
    with open ('run_temp'+ext, 'w') as rsh:
        rsh.write(f'''
{blkm}
{surface}
{filt}
gmt grd2xyz surf_temp.grd > surf_temp.xyz
{pause}
''')

    rsh.close()
        

    os.system( pre+'run_temp'+ext )
    xyz = np.loadtxt('surf_temp.xyz')

    if remove_files == True :
        os.remove('run_temp'+ext )
        os.remove('xyz_temp')
        if os.path.exists("xyz_temp_blkm.txt"):
          os.remove("xyz_temp_blkm.txt")    
        os.remove('surf_temp.grd')
        os.remove('surf_temp.xyz')
        os.remove('gmt.history')
    x = np.unique(xyz[:, 0])
    y = np.unique(xyz[:, 1])
    zz = np.reshape(xyz[:, 2], (len(y), len(x)))
    xx = np.reshape(xyz[:, 0], (len(y), len(x)))
    yy = np.reshape(xyz[:, 1], (len(y), len(x)))
    
    if plot == True:
        plta(zz)
        
    return (xx, yy, zz)


# -----------------------------------------------------------------------------
def grid_fill( data, invalid=None ):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """

    if invalid is None: 
        invalid = np.isnan(data)

    ind = sp.ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    
    return data[ tuple(ind) ]

# -----------------------------------------------------------------------------
def mask_edges( mask, plot=False, c='k', s=None, plt_array=None, vmin=None, vmax=None ) :
    
    """
    Returne x-y coordinates of the edges of a "mask-array" ( boolean or 0/1 array )
    
    0 0 0 0 0 0 0     0 0 0 0 0 0 0     ( (1,1)
    0 1 1 1 1 1 0     0 1 1 1 1 1 0       (1,2)
    0 1 1 1 1 1 0     0 1 0 0 0 1 0       (1,3) 
    0 1 1 1 1 1 0 ==> 0 1 0 0 0 1 0 ==>   (1,4)
    0 1 1 1 1 1 0     0 1 0 0 0 1 0       (1,5)
    0 1 1 1 1 1 0     0 1 1 1 1 1 0       (2,1)
    0 0 0 0 0 0 0     0 0 0 0 0 0 0       (2,5)... )
    """
    
    if type( mask ) == bool :
        mask = mask*1
    
    fil = [[-1,-1,-1],
           [-1, 8,-1],
           [-1,-1,-1]]
    
    edges = np.where( sp.ndimage.convolve(mask*1, fil, mode='constant') > 1)
    
    x_ind, y_ind = edges[1], edges[0] 
        
    if plot is True :
        if plt_array is not None :
            plt.imshow( plt_array, vmin=vmin, vmax=vmax, cmap='rainbow')
            plt.colorbar()
        plt.scatter( x_ind, y_ind, c=c, s=s )
        
    return x_ind, y_ind    

# -----------------------------------------------------------------------------
def normalize( array, b=1, a=0 ) :
    
    Min = np.nanmin(  array )
    Max = np.nanmax( array )
    
    norm = ( b - a ) * ( ( array - Min ) / ( Max - Min ) ) + a
        
    return norm  

# -----------------------------------------------------------------------------
def fillnan( array, xy=None, method='nearest', size=3, iter=1, maxSearchDist=None, 
             plot=False, vmin=None, vmax=None, edges=False, smooth=0, tension=0.2 ) :
    
    zz = np.copy( array )
    
    if np.all( np.isfinite( zz ) ):
        zzfn = zz
    
    if ( array.shape[1] > 1 ) and ( xy is None ):
        xi, yi = np.arange(0, zz.shape[1], 1), np.arange(0, zz.shape[0], 1)
        xx, yy = np.meshgrid( xi, yi )
        
    if xy is not None:
        xx, yy = xy[0], xy[1]
        
    zz_nan = np.isnan( zz )  
    
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
    nan = np.isnan(z)
    notnan = np.invert(nan)
    xn, yn = x[nan], y[nan]
    xi, yi, zi = x[notnan], y[notnan], z[notnan]
    
    if method == 'nearest':
        zfn = sp.interpolate.griddata( ( xi, yi), zi, (xn, yn), method='nearest')
        zn = np.copy(z)
        zn[nan] = zfn
        zzfn = zn.reshape( zz.shape )
        
    if method == 'nearest2D' :
        zzfn = grid_fill( array )
        
    if method == 'mean':
        zfn = np.nanmean( z )
        z[ np.isnan( z ) ] = zfn
        zzfn = z.reshape( zz.shape )
        
    if type( method ) in ('float', 'int'):
        zfn = method
        z[ np.isnan( z ) ] = zfn
        zzfn = z.reshape( np.shape( zz ) )
        
    if method == 'gdal':
        zz[ np.isnan( zz ) ] = 1e-24
        rx, ry, _ = stepxy( xx, yy )
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create( '/vsimem/raster_filled.vrt', zz.shape[1], zz.shape[0],
                                 bands = 1, eType=gdal.GDT_Float32 )
    
        raster.SetGeoTransform( [ xx.min(), rx, 0.0, yy.max(), 0.0, -ry ] )
        raster.GetRasterBand( 1 ).SetNoDataValue( 1e-24 )
        
        out_band = raster.GetRasterBand( 1 )
        out_band.WriteArray( zz )        
        
        if maxSearchDist is None: 
            maxSearchDist = int( np.max( zz.shape ) )
            
        gdal.FillNodata( targetBand = raster.GetRasterBand(1), maskBand=None, 
                         maxSearchDist=maxSearchDist,
                         smoothingIterations=iter )
        
        zzfn = raster.GetRasterBand( 1 ).ReadAsArray()
        zzfn[ zzfn == 1e-24 ] = np.nan
        raster = None
        
    if method in ( 'gauss', 'uniform' ) :
        zn = np.copy( zz ) - np.nanmean( zz )
        zn = grid_fill( zn, invalid=None )
        if method == 'gauss' :
            func = sp.ndimage.gaussian_filter
        if method == 'uniform' :
            func = sp.ndimage.uniform_filter
        
        for i in range( iter ) :
            FG = func( zn, size )
            WF = func( ~zz_nan*1.0, size )
            DFG =  zn - FG
            zn = FG + DFG * WF 
        
        zzfn = zn + np.nanmean( zz )
        
    if method == 'surface':
        zzfn = gmt_surf( xx.ravel(), yy.ravel(), zz.ravel(), grid_shape=zz.shape,
                         tension_factor=tension )[2]
        zzfn=np.flipud( zzfn )      
        
    if edges is True :
        zzfn = filt_edges( zzfn, mask=~zz_nan*1, iter=iter, size=size, smooth=smooth )     
        
    if plot == True:
        plta( array, sbplt=[1, 3, 1], tit='original', vmin=vmin, vmax=vmax )
        plta( zzfn, sbplt=[1, 3, 2], tit='fill', vmin=vmin, vmax=vmax )
        plta( array - zzfn, sbplt=[1, 3, 3], tit='differences' )
        
    return zzfn

# -----------------------------------------------------------------------------
def geo_line_dist( x_line, y_line, prjcode_in=4326, prjcode_out=4326, order=None ) :
    
    if prjcode_in != prjcode_out :
        x, y = prjxy( prjcode_in, prjcode_out, x_line, y_line)
    else :
        x, y = x_line, y_line  

    dist = np.zeros( x.shape )
    
    if order is None :
        xmin, xmax, ymin, ymax = xy2lim( x, y )
        
        x_range = xmax - xmin
        y_range = ymax - ymin
        
        if x_range >= y_range :
            idx0 = x == xmin
        if x_range < y_range :
            idx0 = y == ymin        

        for i, xi in enumerate( x ) :
                dist[i] = np.sqrt( ( x[i]-x[idx0][0] )**2 + ( y[i]-y[idx0][0] )**2   )
        idx_sort = dist.argsort()
        dxy = np.column_stack( ( dist[idx_sort], x[idx_sort], y[idx_sort] ) )                
                
    else :
        if type(order) == str :
            if order == 'same' :
                order = np.arange( x.shape[0] )
        idx_sort = np.argsort( order )
        x, y = x[ idx_sort ], y[ idx_sort ]
        for i, xi in enumerate( x ) :
            if i == 0 : continue 
            dist[i] = dist[i-1] + np.sqrt( ( x[i]-x[i-1] )**2 + ( y[i]-y[i-1])**2 )        
        dist = dist[ np.argsort( idx_sort ) ]
        dxy = np.column_stack( ( dist, x, y ) ) 
        
        
    return dist, idx_sort, dxy   

# -----------------------------------------------------------------------------
def sort_lines( xyzl, prjcode_in=None, prjcode_out=None, add_dist=True,
                line_c=3, x_c=0, y_c=1, order_c=None, plot=False ) :
    
    xyzl = np.column_stack( ( xyzl, np.arange( xyzl.shape[0] ) ) )
    
    idx_sl = np.argsort( ( xyzl[:,line_c] ) )
    
    xyzl = xyzl[ idx_sl ]
    
    lines = np.unique( xyzl[:,line_c] )
    
    new_xyzl = np.zeros( xyzl.shape[1] + 1 )
    
    for i, l in enumerate( lines ) :
        
        idx = xyzl[ :, line_c ] == l 
        
        if order_c is not None :
            order = xyzl[ idx, order_c ]
        else:
            order = None
        
        dist, idx_sort, dxy = geo_line_dist( xyzl[idx, x_c], xyzl[idx, y_c], 
                              prjcode_in=prjcode_in, prjcode_out=prjcode_out, 
                              order=order )    
        
        new_line = np.column_stack( ( xyzl[idx], dist ) )        
        new_xyzl = np.vstack( ( new_xyzl, new_line ) )
        
    new_xyzl = new_xyzl[ 1: , : ]   
    idx_s = np.lexsort( ( new_xyzl[:,-1], new_xyzl[:,line_c] ) )
    new_xyzl = new_xyzl[ idx_s ]
    
    idx_original = np.argsort( new_xyzl[:,-2] )
    new_xyzl = np.delete( new_xyzl, -2, 1 )

    if add_dist != True :
       new_xyzl = new_xyzl[ : , :-1 ] 
    
    if plot is True :
        for l in lines :
            
            idx = new_xyzl[ : , line_c ] == l
            line = new_xyzl[ idx ]
            _ = plt.plot( line[:,x_c], line[:,y_c], c='k' )
    
    return new_xyzl, idx_original   

# -----------------------------------------------------------------------------
def split_lines( xyzl, prjcode_in=None, prjcode_out=None, del_points=True,
            line_c=3, x_c=0, y_c=1, order_c=None, plot=False, new_xy=False,
            angle_th=45, dist_th=None, new_line_c=None ) :
    
    xyzl = np.copy( xyzl )    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                              xyzl[:,x_c], xyzl[:,y_c] )    
    
    if order_c != 'same' :
        xyzl = sort_lines( xyzl, add_dist=False, line_c=line_c, x_c=x_c, y_c=y_c,
                order_c=order_c )[0]
                
    lines = np.unique( xyzl[:,line_c] )
    
    if ( new_line_c is None ) or ( new_line_c is False ) :
        new_line_c = line_c
    elif new_line_c is True :
        new_line_c = -1
        xyzl = np.column_stack( ( xyzl, np.zeros( xyzl.shape[0] ) ) )
 
    xyzl_new = np.empty( ( 0, xyzl.shape[1], ) )
    
    nl = 0
    for il, l in enumerate( lines ) :
        
        nl += 1 
        
        idx = xyzl[ :, line_c ] == l 
        line = xyzl[ idx ]
        line_dist = geo_line_dist( line[:,x_c], line[:,y_c], order='same' )[0] 
                                   
        for ip, p in enumerate( line ) :
            
            p_new = np.copy( p )
            
            if ( dist_th is not None ) and ( ip != 0 ) and \
               ( xyzl_new.shape[0] >=1 ) and ( xyzl_new[-1,new_line_c] == nl ):
                if line_dist[ip] - line_dist[ip-1] > dist_th :
                    nl += 1
                    p_new[ new_line_c ] = nl
                    xyzl_new = np.vstack( ( xyzl_new, p_new ) ) 
                    continue            
            
            if ( ip != 0 ) and ( ip != line.shape[0]-1 ) and \
               ( xyzl_new.shape[0] >= 1 ) and ( xyzl_new[-1,new_line_c] == nl ) :
                    
                vector_1 = [ line[ip-1,x_c]-p[x_c], line[ip-1,y_c]-p[y_c] ]
                vector_2 = [ line[ip+1,x_c]-p[x_c], line[ip+1,y_c]-p[y_c] ]
                unit_vector_1 = vector_1 / ( np.linalg.norm( vector_1 ) + 0.000001 )
                unit_vector_2 = vector_2 / ( np.linalg.norm( vector_2 ) + 0.000001 )
                dot_product = np.dot( unit_vector_1, unit_vector_2 )
                if dot_product <= -1 : dot_product = -0.9999
                if dot_product >= 1 : dot_product = 0.9999
                angle = np.degrees( np.arccos( dot_product ) )
                if angle > 180 :
                    angle = 360 - angle
                angle = 180 - angle 
                if angle >= angle_th :
                    p_new[ new_line_c ] = nl
                    xyzl_new = np.vstack( ( xyzl_new, p_new ) )
                    nl += 1                     
                else :
                    p_new[ new_line_c ] = nl
                    xyzl_new = np.vstack( ( xyzl_new, p_new ) )
            else :
                p_new[ new_line_c ] = nl
                xyzl_new = np.vstack( ( xyzl_new, p_new ) )             
            
    if new_xy is False :
        xyzl_new[:,x_c], xyzl_new[:,y_c] = prjxy( prjcode_out, prjcode_in, 
                xyzl_new[:,x_c], xyzl_new[:,y_c] ) 
        
    if del_points is True :
        del_points = 1
    if type( del_points ) == int :    
        i_p = np.full( xyzl_new.shape[0], True )
        lines = np.unique( xyzl_new[:,line_c] )
        for i, l in enumerate( lines ) :
            idx = xyzl_new[ :, line_c ] == l
            if np.sum( idx ) <= del_points :
                i_p[ idx ] = False
        print( 'Deleted points : ', np.sum( ~i_p ) )    
        xyzl_new = xyzl_new[ i_p ]
    
    if plot is True :
        plt.figure()
        lines = np.unique( xyzl_new[ :, new_line_c ])
        for l in lines :
            
            idx = xyzl_new[ : , new_line_c ] == l
            line = xyzl_new[ idx ]
            plt.plot( line[:,x_c], line[:,y_c], c='k' )
            plt.scatter( line[:,x_c], line[:,y_c], c='r' ) 
                
    return xyzl_new, i_p
        
# -----------------------------------------------------------------------------
def pad_lines( xyzl, pad_dist, pad_idx=-1, method='nearest', prjcode_in=None,
               prjcode_out=None, x_c=0, y_c=1, z_c=2, line_c=3, plot=False, s=1, 
               radius=0, order_c=None, dist=None ) :
    
    xyzl = np.copy( xyzl )
    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c])
    
    xyzl, idx_original = sort_lines( xyzl, add_dist=False, x_c=x_c, y_c=y_c, 
                                     line_c=line_c, order_c=order_c )
    
    xyzli = np.column_stack( ( xyzl, idx_original ) ) 
    
    pad_xyzl = np.zeros( ( 0, xyzli.shape[1] ) ) 
    
    lines_id = np.unique( xyzli[:,line_c] )
    
    for i, l in enumerate( lines_id ) :
        
        line = xyzli[ xyzli[ :, line_c ] == l ] 
        
        len_line = line.shape[0]
        
        if len_line > 1 :
            
            if dist is None :    
                dist = min_dist( line[ :, x_c], line[ :, y_c] )['mean']
            
            if radius is None :
                radius = dist * 2
                
            n = int( int( pad_dist / dist ) + ( pad_dist % dist > 0 ) ) # round Up number of padding points
            new_line = np.copy( line )
            ramp_x = np.arange( 1, n+1 )
#            ramp = ( ramp_x/n )**4 - 2 *( ramp_x/n )**2 + 1
            r=0.1
            if n <= line.shape[0] :
                c = int( n/2 )
            else :
                c = int( line.shape[0] )
            ramp = ( -1 * ( np.e**(c*r) ) + 1 * ( np.e**(r*ramp_x) ) ) / ( ( np.e**(c*r) + ( np.e**(r*ramp_x) ) ) )
                            
            tg_l= []
            
            for i in range( 0, line.shape[0]-1 ) :
                if ( line[ i, y_c ] == line[ i+1, y_c ] ) and  \
                   ( line[ i, x_c ] == line[ i+1, x_c ] ) :
                    continue
                else :
                    num_i = line[ i, y_c ] - line[ i+1, y_c ] 
                    den_i = line[ i, x_c ] - line[ i+1, x_c ] 
                    if den_i == 0 :
                        tg_l.append( np.inf )
                    else :
                        tg_l.append( num_i / den_i ) 
  
            tg_l = np.abs( np.mean(tg_l) )
            
            for i in range( 1, n ) :
                
                signx = np.sign( line[ 0, x_c ] - line[ -1, x_c ] )
                signy = np.sign( line[ 0, y_c ] - line[ -1, y_c ] )
                row_i = np.copy( line[ 0, : ] )
                row_i[ x_c ] = line[ 0, x_c ] + signx * dist * i * np.cos( np.arctan( tg_l ) ) 
                row_i[ y_c ] = line[ 0, y_c ] + signy * dist * i * np.sin( np.arctan( tg_l ) )
                row_i[ -1 ] = pad_idx
                row_i[ z_c ] = line[ 0, z_c ]
                    
                row_f = np.copy( line[ -1, : ] )
                row_f[ x_c ] = line[ -1, x_c ] - signx * dist * i * np.cos( np.arctan( tg_l ) ) 
                row_f[ y_c ] = line[ -1, y_c ] - signy * dist * i * np.sin( np.arctan( tg_l ) )
                row_f[ -1 ] = pad_idx 
                
                row_f[ z_c ] = line[ -1, z_c ]
                new_line = np.vstack( ( row_i, new_line, row_f ) )
                
            if radius != 0 : 
                nearest_i = neighbors_points( ( line[ :, x_c ], line[ :, y_c ] ), 
                              ( line[ 0, x_c ], line[ 0, y_c ] ), radius )[2]
                rampi = normalize( ramp, line[ 0, z_c ], 
                                   np.nanmean( line[ nearest_i, z_c ] ) )
                new_line[0:n,z_c] = rampi
                nearest_f = neighbors_points( ( line[ :, x_c ], line[ :, y_c ] ), 
                              ( line[ -1, x_c ], line[ -1, y_c ] ), radius )[2]
                rampf = normalize( ramp, line[ -1, z_c ], 
                                   np.nanmean( line[ nearest_f, z_c ] ) )
                new_line[-n:,z_c] = np.flipud( rampf )                  
        
        else :
            new_line = line
        
        pad_xyzl = np.vstack( ( pad_xyzl, new_line ) )   
        
    idx_original = pad_xyzl[ :, -1 ]
    pad_xyzl = np.delete( pad_xyzl, -1, 1 )       
    
    if plot == True :
        
        plt.figure()
        
        add_idx = idx_original == pad_idx
        
        plt.scatter( pad_xyzl[~add_idx, x_c], pad_xyzl[~add_idx, y_c], c='k', s=s, label='Original points')
        plt.scatter( pad_xyzl[add_idx, x_c], pad_xyzl[add_idx, y_c], c='r', s=s, label='Added points')
        plt.legend()
        plt.gca().set_aspect('equal')
        
    return pad_xyzl, idx_original  

# -----------------------------------------------------------------------------
def cross_over_points( xyzl, method='nearest', s=2, cmap='rainbow', plot=False, 
                       linewidths=0.05, vmin=None, vmax=None, colorbar=True,
                       prjcode_in=4326, prjcode_out=4326, extend_perc=50,
                       x_c=0, y_c=1, z_c=2, line_c=3, new_xy=False, absolute=True ) :
    
    xyzl = np.copy( xyzl )    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c] )      
    
    lines = np.unique( xyzl[:,line_c] )
    
    cross_lines = []
    cross_points = []
    cross_val = []
    
    for i1, l1 in enumerate( lines ) :
    
        if i1 is len(lines)-1 : 
            break
    
        li = xyzl[ xyzl[ :, line_c ] == l1 ]
        if li.shape[0] <= 1 : continue
    
        col_xy = [ x_c, y_c ]
        lsi = LineString( li[ :, col_xy ] )
            
        limi = xy2lim( li[:,x_c], li[:,y_c], extend=True, d=extend_perc, 
                       method='percentage' )    
        
        _,_,idx = xy_in_lim( xyzl[:,x_c], xyzl[:,y_c], limi )
        possible_cross_lines = np.unique( xyzl[idx, line_c] )
        possible_cross_lines = possible_cross_lines[ possible_cross_lines != l1 ]
        
        if ( np.size( possible_cross_lines ) == 0 ) : continue
            
        for i2, l2 in enumerate( possible_cross_lines ) :
            
            lii = xyzl[ xyzl[ :, line_c ] == l2 ] 
            if lii.shape[0] == 1 : continue
        
            lsii = LineString( lii[ :, col_xy ] )
            
            if lsi.intersects( lsii ) :
                
                i = lsi.intersection(lsii)
                
#                if  isiterable( inter ) is False : 
#                    i = [inter]
                
#                for i in inter :
                if type( i ) == shapely.geometry.point.Point :
                    if ( i.x, i.y ) not in cross_points :
                            
                        cval_i = xyz2xy( ( li[:,x_c], li[:,y_c], li[:,z_c] ), 
                                         ( i.x, i.y ), method=method  )[0]  
                        cval_ii = xyz2xy( ( lii[:,x_c], lii[:,y_c], lii[:,z_c] ), 
                                          ( i.x, i.y ), method=method )[0] 
                        c_diff = cval_i - cval_ii
                        if absolute == True :
                            c_diff = np.abs( c_diff ) 
                    
                        cross_points.append( ( i.x, i.y ) )
                        cross_lines.append( ( l1, l2 ) )
                        cross_val.append( ( cval_i, cval_ii, c_diff ) )
                else :
                    continue
                        
    cross_points = np.array( cross_points )
    cross_lines = np.array( cross_lines )
    cross_val = np.array( cross_val ) 
    
    cross_over_array = np.column_stack( ( cross_points, cross_lines, cross_val ) )
    
    if plot == True :
        
        plt.scatter( xyzl[:,x_c], xyzl[:,y_c], s=s, marker='_', c='k', 
                     linewidths=linewidths, alpha=0.5 )
        
        plt.scatter( cross_over_array[:,0], cross_over_array[:,1], s=s*10, 
                     c=cross_over_array[:,6], cmap=cmap, vmin=vmin, vmax=vmax )
        
        if colorbar is True :
            plt.colorbar()
        
        Min, Max, Mean, Std = stat( cross_over_array[:,6], decimals=2 )
        
        tit = f'Cross-Over Error : \n' + f' Min={Min}  Max={Max}  Mean={Mean}  Std={Std} '
        plt.title( tit )
        
    if new_xy is False :
        cross_over_array[:,0], cross_over_array[:,1] = prjxy( prjcode_out, prjcode_in, 
                cross_over_array[:,0], cross_over_array[:,1] )         
        
    return cross_over_array

# -----------------------------------------------------------------------------
def block_m( x, y, z, wind_size, method='mean', data_type='vector', lim=None,  
             prjcode_in=4326, prjcode_out=4326, nan=False, adjst_lim=True, 
             plot=False, s1=None, s2=None ) :         
    """
    Block Mean or Block Median
    """

    if prjcode_in != prjcode_out:
        x, y = prjxy(prjcode_in, prjcode_out, x, y)
    
    if lim == None:
        lim = [np.min(x), np.max(x), np.min(y), np.max(y)]
        
#    x_blk = np.linspace(lim[0], lim[1], int( abs( lim[0]-lim[1] ) / wind_size ) )
#    y_blk = np.linspace(lim[2], lim[3], int( abs( lim[2]-lim[3] ) / wind_size ) )          
         
    x_blk = np.arange(lim[0], lim[1], wind_size)
    y_blk = np.arange(lim[2], lim[3], wind_size) 

    Xb, Yb = np.meshgrid(x_blk, y_blk)
    xb, yb, zb = np.hsplit( np.zeros( ( (Xb.shape[0]-1) * (Xb.shape[1]-1), 3 ) ), 3 )
    
    n = 0
    for idx, _ in np.ndenumerate( Xb ) :
     
        i, j = idx
        if ( i == Xb.shape[0]-1 ) or ( j == Xb.shape[1]-1 ) :
            continue
        
        win_idx = ( ( x > Xb[ ( i, j ) ] ) & ( x < Xb[ ( i, j + 1 ) ]) & \
                    ( y > Yb[ ( i, j ) ] ) & ( y < Yb[ ( i + 1, j ) ] ) )
        
        if method == 'mean':              
            xb[n] = np.mean( x[ win_idx ] )
            yb[n] = np.mean( y[ win_idx ] )
            zb[n] = np.mean( z[ win_idx ] )
            
        if method == 'median':              
            xb[n] = np.median( x[ win_idx ] )
            yb[n] = np.median( y[ win_idx ] )
            zb[n] = np.median( z[ win_idx ] )
            
        n += 1    

    if data_type == 'grid' :
        if adjst_lim == True :
            x_grid = np.arange(np.min(xb), np.max(xb)+wind_size, wind_size)
            y_grid = np.arange(np.min(yb), np.max(yb)+wind_size, wind_size)
        if adjst_lim == False :
            x_grid = np.linspace(np.min(xb), np.max(xb), int( ( np.max(xb) - np.min(xb) ) / wind_size ) )
            y_grid = np.linspace(np.min(yb), np.max(yb), int( ( np.max(yb) - np.min(yb) ) / wind_size ) )
        Xg, Yg = np.meshgrid( x_grid, y_grid )
        Zg = xyz2xy( ( xb.ravel(), yb.ravel(), zb.ravel() ), ( Xg, Yg ), method='linear' )[0]
        xb, yb, zb = Xg, Yg, Zg
        if ( y[0,0] - y[-1,0] ) * ( yb[0,0] - yb[-1,0] ) < 0 :
            yb = np.flipud( yb )
            zb = np.flipud( zb )
        
    if data_type == 'vector' :
        if nan == False :
            xb = xb[ ( ~np.isnan( zb ) ) ]
            yb = yb[ ( ~np.isnan( zb ) ) ]
            zb = zb[ ( ~np.isnan( zb ) ) ]
    
    if plot == True:
        plt.figure()
        plt.scatter(x, y, c='b', s=s1)
        plt.scatter(xb, yb, c='r', s=s2)  
        
    return xb, yb, zb

# -----------------------------------------------------------------------------
def resamp_lines( xyzl, dist, method='nearest', prjcode_in=4326, prjcode_out=4326, 
                  cmap='rainbow', plot=False, x_c=0, y_c=1, z_c=2, line_c=3, s=1,
                  order_c=None, new_xy=False, lines=[] ) : 

    xyzl = np.copy( xyzl )
    if prjcode_in != prjcode_out:
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c] ) 

    if order_c != 'same' :
        xyzl = sort_lines( xyzl, x_c=x_c, y_c=y_c, line_c=line_c, order_c=order_c )[0]    

    xyzl_new = np.empty( ( 0, 4, ) )
    
    if lines == [] :
        lines = np.unique( xyzl[ :, line_c ] ) 

    for l in np.unique( xyzl[ :, line_c ].astype(int) ) :
        
        li = xyzl[ xyzl[ :, line_c ] == l ]
        
        if ( li.shape[0] > 1 ) and ( l in lines ) : 
            
            col_xy = [ x_c, y_c ]
            lsi = LineString( li[ :, col_xy ] )
            line_dist = geo_line_dist( li[:,x_c], li[:,y_c], order='same' )[0]
            
            if line_dist[-1] < dist : 
                xyzl_i = np.array( ( [ li[:,x_c][0],li[:,y_c][0],li[:,z_c][0],li[:,line_c][0] ],
                                     [ li[:,x_c][-1],li[:,y_c][-1],li[:,z_c][-1],li[:,line_c][-1] ] ) )
                xyzl_new = np.vstack( ( xyzl_new, xyzl_i ) ) 
            else :    
                line_dist_new = np.linspace( 0, line_dist.max()+dist, int( line_dist.max() / dist) )
                
                xy = []
                for d in line_dist_new :
                    xn, yn = lsi.interpolate( d ).xy
                    xy.append( [ xn[0], yn[0] ] )
                
                xy = np.array( xy )
                l_rep = np.repeat( l, xy.shape[0] )
                z_new = np.zeros( xy.shape[0] )
                
                xyzl_i = np.column_stack( ( xy[:,0], xy[:,1], z_new, l_rep ) )
                x_start, y_start = li[:,x_c][0], li[:,y_c][0]  
                
                dist_xy =  np.sqrt( ( xy[:,0] - x_start )**2 + ( xy[:,1] - y_start )**2 )
                    
                un_line_dists, idx = np.unique( line_dist, return_index=True )  
                un_z_line = li[:,z_c][idx]                 
                int_c_f = sp.interpolate.interp1d( un_line_dists, un_z_line,
                                                   kind=method, fill_value='extrapolate' )
                xyzl_i[ :, 2 ] = int_c_f( dist_xy )        
                
                xyzl_new = np.vstack( ( xyzl_new, xyzl_i ) ) 
            
        else :
            xyzl_i = np.column_stack( ( li[:,x_c], li[:,y_c], li[:,z_c], li[:,line_c] ) )
            xyzl_new = np.vstack( ( xyzl_new, xyzl_i ) )            
        
    if plot is True :
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.scatter( xyzl[:, x_c], xyzl[:, y_c], c=xyzl[:, z_c], s=s, 
                     label='Original', cmap=cmap )
        plt.legend()
        plt.gca().set_aspect('equal') 
        plt.subplot(1,2,2)
        plt.scatter( xyzl_new[:, 0], xyzl_new[:, 1], c=xyzl_new[:, 2], s=s, 
                     label='Resampled', cmap=cmap )
        plt.legend()
        plt.gca().set_aspect('equal') 
        
    if new_xy is False :
        xyzl_new[:,0], xyzl_new[:,1] = prjxy( prjcode_out, prjcode_in, 
                xyzl_new[:,0], xyzl_new[:,1] )         
        
    return xyzl_new    
  
# -----------------------------------------------------------------------------
def lines_samp_dist( xyzl, prjcode_in=4326, prjcode_out=4326, kind='mean',
                     x_c=0, y_c=1, line_c=3, s=1, round_dist=6, print_dist=True,
                     deg_to_m=False ) : 
                       
    if prjcode_in != prjcode_out:
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c] )   

    lines = np.unique( xyzl[ :, line_c ]  )    
    
    lines_dist = []
    
    for l in lines :
        
        li = xyzl[ xyzl[ :, line_c ] == l ]
        lines_dist.append( min_dist( li[:,x_c], li[:,y_c] )[kind] )
        
        
    minimum_dist = round( np.nanmean( lines_dist ), round_dist )
    
    if deg_to_m is True :
        print( 'Minimum sampling distance :', deg2m( minimum_dist ) )

    else :
        print( 'Minimum sampling distance :', minimum_dist )
    
    return minimum_dist

# -----------------------------------------------------------------------------
def array2csv( array, headers=[], sep=',', fmt='% 15.6f', 
               path_name='new_dataset', nan=None, nan_c=2 ) :

    path = os.path.dirname( path_name )
    print(path)
    if path != '' :
        os.makedirs( path, exist_ok=True )
    
    if nan != None :
        inan = np.isnan( array[ :, nan_c] )
        if ( type( nan ) == bool ) and ( nan == False ) :
            array = array[ ~inan ]
        if type( nan ) in ( float, int ) :
            array[ inan, nan_c ] = nan
            
    hd = ''
    fmt_str = ''
    n_col = array.shape[1]
    
    if type( fmt ) == str :
        fmt = [fmt]
        fmt = [ fmt[0] for i in range( n_col ) ]
        
    for i, _ in enumerate( fmt ):
        fmt_str = fmt_str + fmt[i] + sep
    fmt_str = fmt_str[:-1]
        
    if headers != []:        
        for i, h in enumerate( headers ):
            space = fmt[i]
            for ch in ['% ', 'f', 'd', 'i', 'e', '> ' ] :
                space = space.replace(ch, '.')    
            space = space.split('.')[1]
            hd = hd + f'{h:>{space}}' + sep
        hd = hd[:-1]   
     
    np.savetxt( path_name, array, header=hd, fmt=fmt_str, 
                comments='', delimiter=sep )
    
    abs_path = os.path.abspath(path_name)

    return abs_path

# -----------------------------------------------------------------------------
def read_csv( csv_file, sep=' ', header=None, skiprows=[], skipcols=[], fmt=[],
              encoding="utf8", n_col=None, adjust_last_col_fmt=True, comments=[],
              col_size=[], new_sep=',', printf=False, decimals=2, space=12,
              rows=[], cols=[], show=False, nan=np.nan ) :
    
    dictionary = {}
    
    f = open( csv_file, "r", encoding=encoding, errors='ignore' )
    
    lines = f.readlines()
    
    if col_size != [] :

        for i, l in enumerate( lines ) :
            
            n = 0
            c = 0
            for sz in col_size :
                
                c += sz 
                
                l = l[:c+n] + new_sep + l[c+n:]
                n += 1
                lines[ i ] = l
                
        n_col = n
        sep = new_sep
    
    if printf is True :
        
        if rows == []:
            
            rows = range( len(lines) )
            
        for i in rows :
            
            print( lines[i], end="" )
            
    data = None
    hd = None
    
    if comments != [] :
        
        if skiprows == [] :
            
            skiprows = comments  
            
        comments = [ lines[i] for i in comments ]
        
    if (type(header) == int) and (header > 0) :
        
        skiprows = skiprows + [n for n in range(header)]
        
    if type( skiprows ) == int :
        
        skiprows = [n for n in range(skiprows)]
        
    for i, l in enumerate( lines ) :   
        
        if ( i in skiprows ) :
            
            continue
        
        if skipcols != [] :

            for i, li in enumerate(l) :

                l = np.delete( l, skipcols ).tolisit()
                    
        if i == header :
            
            hd = [ h.strip() for h in l.split( sep ) if ( (h!='') and (h.strip()!='') ) ]
            
            continue
        
        if hd != None :
            
            n_col = len( hd )
            
        if ( l != "\n" ) and ( data == None ) and ( n_col == None ) :
            
            n_col = len( [ n.strip() for n in l.split( sep ) if ( (n!='') and (n.strip()!='') ) ] )
            
        if ( n_col != None ) and ( data == None ) :
            
            data = [ [] for i in range( n_col ) ]
        
        if ( l != "\n" ) and ( data != None ) :
            
            h_fmt = l.split( sep )
            
            if col_size != [] :
                
                line_list = [ n.strip() for n in l.split( sep ) ]
            
            else :
                
                line_list = [ n.strip() for n in l.split( sep ) if ( (n!='') and (n.strip()!='') ) ]
                    
            for c in range( n_col ) :
                
                val = line_list[c]
                
                if ( val == '' ) or ( val.strip() == '' ) :
                    val = nan     
                data[c].append( val )  
    f.close() 
    
    
    if hd == None :
        
        hd = [ 'c' + str(n+1) for n in range( n_col) ]
        
    for i, h in enumerate( hd ) :
        
        try:
            
            dictionary[h] = np.asarray( data[i], dtype=float )
            
        except ValueError:
            
            dictionary[h] = np.array( data[i] )    
            
    array = dict2array( dictionary )[0]   
    
    fmt = []
    hfmt = []
    beg = 0
    for i, j in enumerate(h_fmt) :
        
        if i != 0 :
            
            if ( i!= len(h_fmt)-1 ) and ( j == '' ) and ( h_fmt[i-1] != '' ) :
                
                hfmt.append( ' '.join(h_fmt[ beg:i ]) )
                
                beg=i
                
            if ( i == len(h_fmt)-1 ) and ( beg!= 0 ) :
                
                hfmt.append( ' '.join(h_fmt[ beg: ] ) )
                
    if hfmt == [] :
        
        hfmt = h_fmt            
                    
    for i, k in enumerate( hfmt ) :  
        
        if ( i == len( hfmt ) - 1 ) and ( adjust_last_col_fmt == True ) : 
            len_l = str( len( k ) -1 )
        
        else :    
            len_l = str( len( k ) )
        string_str = k.strip()
        
        try:
            _ = int( string_str )
            fmt.append('% '+len_l+'d')
        except:
            
            try :
                _ = float( string_str )
                ndec = str( len( string_str.split( '.' )[1] ) )
                fmt.append('% '+len_l+'.'+ndec+'f')
            
            except :
                fmt.append('% '+len_l+'s')  

    if header == None:
        hd = None 
        
    if show is True :
        _ = print_table( dictionary, decimals=decimals, space=space, rows=rows,
                          cols=cols )            
        
    return dictionary, array, hd, fmt, comments, data     

# -----------------------------------------------------------------------------
def csv2dict( csv_file, sep=',', header=None, skiprows=[], encoding="utf8", 
              n_col=None, adjust_last_col_fmt=True ) :
    
    d = read_csv( csv_file, sep=sep, header=header, skiprows=skiprows, 
                  encoding=encoding, n_col=n_col, 
                  adjust_last_col_fmt=adjust_last_col_fmt )[0]

    return d

# -----------------------------------------------------------------------------
def csv2array( csv_file, sep=',', header=None, skiprows=[], encoding="utf8", 
               n_col=None, adjust_last_col_fmt=True ) :
    
    a = read_csv( csv_file, sep=sep, header=header, skiprows=skiprows, 
                  encoding=encoding, n_col=n_col, 
                  adjust_last_col_fmt=adjust_last_col_fmt )[1]

    return a
   
# ----------------------------------------------------------------------------- 
def dict2array( dictionary, show=False, exclude=[], rows=[], cols=[] ) :
    
    headers = []
    
    for i, k in enumerate( dictionary ) :
        
        if k in exclude :
            continue
        if i == 0 :
            array = np.copy( dictionary[k] )  
        else : 
            array = np.column_stack( ( array, dictionary[k] ) )  
        
        headers.append( k )  
        
    # print_table( array, headers=headers, space=12, decimals=2, rows=rows, cols=cols )
    
    return array, headers
            
# -----------------------------------------------------------------------------            
def dict2csv( dictionary, sep=',', fmt='% 15.6f', path_name='new_dataset' ) :
    
    array, headers = dict2array( dictionary )
    
    abs_path = array2csv( array, headers=headers, sep=sep, fmt=fmt, 
                          path_name=path_name )
    
    return abs_path

# -----------------------------------------------------------------------------            
def addc2csv( csv_file, new_c=[], new_hd=[], new_fmt=[], sep=',', rep_fmt=None,
              path_name=None, rep_headers=None, header=None, skiprows=[], 
              encoding="utf8", n_col=None, adjust_last_col_fmt=True ) :
    
    if path_name == None :
        path_name = csv_file
    
    csvf = read_csv( csv_file, sep=sep, header=header, skiprows=skiprows, 
                     encoding=encoding, n_col=n_col, 
                     adjust_last_col_fmt=adjust_last_col_fmt )
    
    array = csvf[1]
    headers = csvf[2]
    fmt = csvf[3]
    n_c = array.shape[1]     
    new_array = np.copy( array )
    if new_hd == [] :
       new_hd = [ 'c_'+str(n_c+i) for i in range( len( new_c ) ) ] 
    if new_fmt == [] :
       new_fmt = [ '% 15.6f' for i in range( len( new_c ) ) ]   
    if type( new_hd ) == str :
        new_hd = [ new_hd ] * len( new_c )    
    if type( new_fmt ) == str :
        new_fmt = [ new_fmt ] * len( new_c )     
   
    for i, col in enumerate( new_c ) :
        new_array = np.column_stack( ( new_array, col ) )
        if headers != None :
            headers.append( new_hd[i] )
        fmt.append( new_fmt[i] )

    if rep_fmt != None :
        fmt = rep_fmt
        
    if rep_headers != None :
        headers = rep_headers

    abs_path = array2csv( new_array, headers=headers, sep=sep, fmt=fmt, 
                          path_name=path_name )
    
    return abs_path
   
# ----------------------------------------------------------------------------- 
class SelectFromCollection(object):
    # Select indices from a matplotlib collection using `LassoSelector`.
    #
    # Selected indices are saved in the `ind` attribute. This tool fades out the
    # points that are not part of the selection (i.e., reduces their alpha
    # values). If your collection has alpha < 1, this tool will permanently
    # alter the alpha values.
    #
    # Note that this tool selects collection objects based on their *origins*
    # (i.e., `offsets`).
    #
    # Parameters
    # ----------
    # ax : :class:`~matplotlib.axes.Axes`
    #     Axes to interact with.
    #
    # collection : :class:`matplotlib.collections.Collection` subclass
    #     Collection you want to select from.
    #
    # alpha_other : 0 <= float <= 1
    #     To highlight a selection, this tool sets all selected points to an
    #     alpha value of 1 and non-selected points to `alpha_other`.
    #

    def __init__(self, ax, collection, alpha_other=0.3, facecolors=None):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        if facecolors is not None: self.fc = facecolors

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
#------------------------------------------------------------------------------
def merge2Darrays( array_list, res_list, sigmab=1, spl_order=[], mean_shift=False,
                   plot=False, vmin=None, vmax=None, buffer=None, plot_diff=False,
                   s=None, fill='nearest2D', iter=None ) :
                   
    Z1 =  np.copy( array_list[0] )
    shape1 = Z1.shape
    if spl_order == [] :
        spl_order = [ 1 for i in array_list ]
    
    for i, array in enumerate( array_list ) :
        
        if i == 0 : continue
        
        Z2 = np.array( array )
        shape2 = Z2.shape
        shape_ratio = shape1[0]/shape2[0], shape1[1]/shape2[1]
        res_ratio = round( res_list[i] / res_list[i-1] ) 
        if buffer is None :
            buffer = int( res_ratio )
            
        buffer = buffer * sigmab
        
        print( 'buffer: ', buffer )
        if (shape_ratio[0] != 1.0) and (shape_ratio[1] != 1.0) :
            Z21 = sp.ndimage.zoom( Z2, shape_ratio, order=spl_order[i] ) 
        else: 
            Z21 = np.copy( Z2 ) 
        
        mask1 = np.isfinite( Z1 ) 
        mask2 = np.isfinite( Z21 )
        mask1n2 = mask1 & ~mask2    
        
        if mean_shift == True:
            print( 'mean_shift :', np.nanmean( Z21[mask1] - Z1[mask1] ) )
            Z1 = Z1 + np.nanmean( Z21[mask1] - Z1[mask1] )        
        # ---
        # If the i-th raster has the original resolution lower than the final chosen resolution,
        # it will be smoothed with a rolling average, 
        # The kernel size of the convolution is equal to the ratio betwee original and final resolution (+1 if it's even).
        # This will reduce aliasing artefacts in the low resolution area  
        
        if np.mean(shape_ratio) > 1 :
#            Z21[ ~mask2 ] = 0 
            Z21 = fillnan( Z21, method=fill )
            if np.mean(shape_ratio) % 2 == 0 :
                Z21 = sp.ndimage.uniform_filter( Z21 , np.mean(shape_ratio)+1 )
            else :
                Z21 = sp.ndimage.uniform_filter( Z21 , np.mean(shape_ratio) )
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
        mask3 = ~sp.ndimage.uniform_filter( ~mask1*1, buffer*2+1 ).astype(bool)
        weight = sp.ndimage.gaussian_filter( mask3*1.0, buffer*2 ) 
        weight[ mask1n2 ] = 1
        D_21[ ~mask2  ] = 0

#        mask1 = weight >= 0.9999
        
        if iter is None :
            iter = int( buffer )
            
#        D_21_1 = std_filt( D_21, radius=1 )
#        WF = np.abs( sp.ndimage.uniform_filter( ~mask1*1.0, 3 ) - 1 )
#        DFG =  D_21 - D_21_1
#        D_21 = D_21_1 + DFG * WF        
            
        D_21fn = fillnan( D_21, method=fill, maxSearchDist=None, iter=iter, edges=True )
        
        FG = np.copy( D_21fn )

        for i in range( iter ) :
            
            FG1 = sp.ndimage.gaussian_filter( FG, 2 )
            WF = np.abs( sp.ndimage.gaussian_filter( ~mask1*1.0, 2 ) - 1 )
            DFG =  FG - FG1
            FG = FG1 + DFG * WF  
            
        FG = FG * weight
#        FG[mask1] = D_21[mask1] 
        
        if plot_diff == True :
            plta( weight, vmin=vmin, vmax=vmax, sbplt=[1,2,1] )
            mask_edges( mask1, plot=True, c='k', s=s )
            plta( FG, vmin=vmin, vmax=vmax, sbplt=[1,2,2] )
            mask_edges( mask1, plot=True, c='k', s=s )        

        FG[ mask1n2 ]  = Z1[ mask1n2 ] 
        
        Z2n = Z21 + FG
        Z2n[ ~( mask1 | mask2 ) ] = np.nan
        Z1 = Z2n
        
        mask5 = ~sp.ndimage.uniform_filter( ~mask1*1, int(buffer) ).astype(bool)
        mask6 = sp.ndimage.uniform_filter( mask1*1, int(buffer) ).astype(bool)
        mask7 = ( mask5 ^ mask6 ) & mask2
        WF = np.abs( sp.ndimage.gaussian_filter( mask7*1.0, 3 ) - 1 )
        plta(WF)
        fw = np.ones( ( 3, 3) ) / 9
        Zf = grid_fill( Z1, invalid=None )
        
        for i in range( buffer ) :
            Z1F = signal.convolve2d( Zf, fw, mode='same' ) 
#            Z1F = sp.ndimage.median_filter( Zf, 1 ) 
            DZ =  Zf - Z1F
            Zf = Z1F + DZ * WF  
        Zf[ ~( mask1 | mask2 ) ] = np.nan
        Z1 = Zf
        
    if plot is True :
        plta( Z1, vmin=vmin, vmax=vmax)

    return Z1

# -----------------------------------------------------------------------------
def rolling_win_2d( arr, win_shape ) :
    
    if ( win_shape[0] % 2 == 0 ) or ( win_shape[1] % 2 == 0 ) :
        raise NameError('Both values in win_shape must be odd integers !')
    
    r_extra = np.floor(win_shape[0] / 2).astype(int)
    c_extra = np.floor(win_shape[1] / 2).astype(int)
    a = np.empty((arr.shape[0] + 2 * r_extra, arr.shape[1] + 2 * c_extra))
    a[:] = np.nan
    a[r_extra:-r_extra, c_extra:-c_extra] = arr    
    
    s = (a.shape[0] - win_shape[0] + 1,) + (a.shape[1] - win_shape[1] + 1,) + win_shape
    strides = a.strides + a.strides
    
    windows = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)
    windows = windows.reshape( arr.size, win_shape[0] * win_shape[1] )
    
    return windows    

# -----------------------------------------------------------------------------
def median_filt( array, radius=1, padw=0, pmode='linear_ramp', 
                 plot=False, vmin=None, vmax=None, iter=1 ):
    
    ar_pad, original_shape_indx = pad_array(array, padw, pmode)
    fw = np.ones( ( radius * 2 + 1, radius * 2 + 1 ) )
    
    for i in range( iter ) : 
        ar_pad = sp.ndimage.median_filter( ar_pad, footprint=fw, 
                                               mode='nearest' )
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt

# -----------------------------------------------------------------------------
def mean_filt( array, radius=1, padw=0, pmode='linear_ramp', 
               plot=False, vmin=None, vmax=None, iter=1 ) :
    
    ar_pad, original_shape_indx = pad_array(array, padw, pmode)
    fw = np.ones( ( radius*2 + 1, radius*2 + 1) ) / ( radius * 2 + 1 ) ** 2
    
    for i in range( iter ) : 
        ar_pad = signal.convolve2d( ar_pad, fw, mode='same' )
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt

#------------------------------------------------------------------------------
def hanning_filt( array, padw=0, pmode='linear_ramp', 
                  plot=False, vmin=None, vmax=None, iter=1 ) :
    
    ar_pad, original_shape_indx = pad_array(array, padw, pmode)
    fw = np.array( [ [ 0, 1, 0 ], [ 1, 2, 1 ], [ 0, 1, 0 ] ] ) / 6
    
    for i in range( iter ) : 
        ar_pad = signal.convolve2d( ar_pad, fw, mode='same' )
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt


# -----------------------------------------------------------------------------
def gauss_filt( array, radius=1, sigma=1, padw=0, pmode='linear_ramp', 
                alpha=None, iter=1, plot=False, vmin=None, vmax=None ):
    
    ar_pad, original_shape_indx = pad_array(array, padw=padw, mode=pmode, alpha=alpha)
    
    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
#    normal = 1 / (2.0 * np.pi * sigma ** 2)
#    fw = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    if radius is not None :
        w = radius * 2 + 1
        truncate = ( ( ( w - 1 ) / 2 ) - 0.5 ) / sigma
    else :
        truncate = 4
        
    for i in range( iter ) : 
#        ar_pad = signal.convolve2d(ar_pad, fw, mode='same')
        ar_pad = sp.ndimage.gaussian_filter(ar_pad, sigma, truncate=truncate)
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt

# -----------------------------------------------------------------------------
def resampling_filt( array, factor, padw=0, pmode='gdal', 
                     alpha=None, spl_order=2, plot=False, vmin=None, vmax=None ) :
    
    if padw == 0 :
        padw = [ 0, 0 ]
    if array.shape[0] % 2 == 0  :
        padw[0] += 1 
    if array.shape[1] % 2 == 0  :  
        padw[1] += 1 

    ar_pad, original_shape_indx = pad_array( array, padw=padw, mode=pmode, alpha=alpha, ptype='' )
    
    # Down_Sampling
    ar_pad_dw = resampling( ar_pad, 1/factor, spl_order=spl_order )
    
    # Upsampling
    ar_pad_up = resampling( ar_pad_dw, factor, spl_order=spl_order )   
    
    ar_filt = crop_pad( ar_pad_up, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt    

# -----------------------------------------------------------------------------
def std_filt( array, padw=0, pmode='gdal', radius=1, n=1, 
              alpha=None, plot=False, vmin=None, vmax=None ) :
    
    ar_pad, original_shape_indx = pad_array(array, padw=padw, mode=pmode, alpha=alpha)
    
    win_shape = radius * 2 + 1, radius * 2 + 1 
    windows = rolling_win_2d( ar_pad, win_shape )
    w_mean = np.nanmean( windows, axis=1 )
    w_std = np.nanstd( windows, axis=1 )
    w_filt = w_mean + np.sign( w_mean ) * w_std
    ar_pad_filt = w_filt.reshape( ar_pad.shape )
    
    ar_filt = crop_pad( ar_pad_filt, original_shape_indx )    
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt        

#------------------------------------------------------------------------------
def filt2d( array, radius=1, padw=0, pmode='linear_ramp', plot=False, vmin=None, 
            vmax=None, iter=1, ftype='mean', sigma=1, factor=2, mask=None,
            fill=None ):

    if type( ftype ) is str :
        ftype = [ ftype ]
    
    if  type( iter ) is int :
        iter = [ iter for ft in ftype ]
        
    if fill != None :
        nan = np.isnan( array )
        array = fillnan( array, method=fill )
    
    ar_filt = np.copy( array )
    for i, f in enumerate( ftype ) :
        
        if ftype[i] == 'mean' :
            ar_filt = mean_filt( ar_filt, radius=radius, padw=padw, pmode=pmode, 
                                 iter=iter[i] )  
        if ftype[i] == 'hanning' :
            ar_filt = hanning_filt( ar_filt, padw=padw, pmode=pmode, iter=iter[i] )
            
        if ftype[i] == 'median' :
            ar_filt = median_filt( ar_filt, radius=radius, padw=padw, pmode=pmode, 
                                   iter=iter[i] )      
        if ftype[i] == 'gauss' :
            ar_filt = gauss_filt( ar_filt, radius=radius, padw=padw, pmode=pmode, 
                                   iter=iter[i], sigma=sigma ) 
        if ftype[i] == 'resamplig' :
            ar_filt = resampling_filt( ar_filt, factor=factor, padw=padw, pmode=pmode ) 
            
    if fill != None :
          ar_filt[ nan ] = np.nan   
            
    if mask is not None:
        ar_filt[ mask ] = np.nan
        array = np.copy(array)
        array[ mask ] = np.nan            

    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt 

# -----------------------------------------------------------------------------
def filt_edges( array, mask=None, size=3, iter=1, plot=False, vmin=None, vmax=None,
                smooth=0 ) :
    
    if mask is None :
        mask = np.isfinite( array )
    
    edge_mask1 = sp.ndimage.uniform_filter( mask*1, size=size ) 
    edge_mask2 = np.abs( sp.ndimage.uniform_filter( 
                         np.abs(mask-1)*1, size=size ) -1 )
    
    edge_mask = edge_mask2 - edge_mask1
    
    edge_mask = sp.ndimage.uniform_filter( edge_mask*1.0, size=size )
    edge_mask[ edge_mask<1e-10 ] = 0
    edge_mask[ edge_mask>1 ] = 1
    
    ar_filt = np.copy( array )
    
    for i in range( iter ) :
        FG = sp.ndimage.uniform_filter( ar_filt, size )
        DFG =  ar_filt - FG
        ar_filt = ar_filt - DFG * edge_mask 
        
    if smooth != 0 :
        mean = np.nanmean( ar_filt )
        edge_mask3, original_shape_indx = pad_array( mask, padw=[2,2], 
                            ptype='', mode='constant', constant_values=0 ) 
        edge_mask3 = edge_mask3 * 2.0
        edge_mask3[:,[0,-1]] = 1.0
        edge_mask3[[0,-1],:] = 1.0   
        edge_mask3[(edge_mask3!=1.0)&( edge_mask3!=2.0)] = 1.5        
        SE = sp.ndimage.gaussian_filter( edge_mask3, smooth ) - 1
        SE = crop_pad( SE, original_shape_indx )
        ar_filt = ( ar_filt - mean ) * SE + mean
        
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
    
    return ar_filt   

# -----------------------------------------------------------------------------
def lim2grid( xyz, lim, step=None, plot=False, vmin=None, vmax=None, prjcode_in=4326,
              prjcode_out=4326, method='linear', blkm=False, filt=False, radius=None,
              nan=True, padw=0 ) :
    
#    lim = copy.copy(lim)
    
    if len( xyz ) <= 2 :
        x, y = xyz
    else :
        x, y, z = xyz
    
    if prjcode_in != prjcode_out :
        x, y = prjxy( prjcode_in, prjcode_out,  x, y )
        
    x, y, idx = xy_in_lim( x, y, lim )
    if len( xyz ) > 2 :
        z = z[idx]

    if step == None :
        step = min_dist( x, y )['mean']
        
    if ( blkm == True ) and  ( len( xyz ) > 2 ) :
        x, y, z = block_m( x, y, z, step*2, method='mean', data_type='vector', lim=lim )  
        
    xg = np.linspace( lim[0], lim[1], int( (lim[1]-lim[0])/step ) ) 
    yg = np.linspace( lim[3], lim[2], int( (lim[3]-lim[2])/step ) )
    X, Y = np.meshgrid( xg, yg )
    
    if len( xyz ) > 2 :
        Z = xyz2xy( ( x, y, z ), ( X, Y ), method=method, fillnan=False )[0]
        
        if filt == True :
            if radius == None :
                radius = int( min_dist( x, y )['mean'] / step )
            Z = filt2d( Z, radius=radius, ftype='mean', padw=padw  )    
        
    if plot == True :
        plta( Z )
        
    if len( xyz ) <= 2 :
        
        return X, Y
    
    if len( xyz ) > 2 :
        
        if nan == False :
            X = X[ ~np.isnan( Z ) ]
            Y = Y[ ~np.isnan( Z ) ]
            Z = Z[ ~np.isnan( Z ) ]
            
        return X, Y, Z
    
# -----------------------------------------------------------------------------
def mask2D( xyr=None, xyzgrid=None, array=None, array_and_lim=None, mask=None,
            prjcode=4326, plot=False, vmin=None, vmax=None, s=0,
            convexhull=False, cut2edges=False, in_out='in' ):

    if xyr is not None:
        x, y, r = xyr
        
    if xyzgrid is not None:
        xx, yy, zz = xyzgrid[0], xyzgrid[1], xyzgrid[2]
        xa, ya, za = xx.ravel(), yy.ravel(), zz.ravel()
        
    if array is not None :
        zz = np.copy( array )
        x, y = np.arange( zz.shape[1] ), np.arange( zz.shape[0] )
        xx, yy = np.meshgrid( x, y )
        xa, ya, za = xx.ravel(), yy.ravel(), zz.ravel()
        
    if array_and_lim is not None: 
        zz = array_and_lim[0]
        lim = array_and_lim[1]
        nx, ny = zz.shape[1], zz.shape[0]
        xi, yi = np.linspace(lim[0], lim[1], nx), np.linspace(lim[3], lim[2], ny)
        xx, yy = np.meshgrid(xi, yi)
        xa, ya, za = xx.ravel(), yy.ravel(), zz.ravel()
        
    if ( xyr is not None ) and ( convexhull is False ) :
        zm = np.empty( np.shape( za ) )
        zm[:] = np.nan
        for p in zip( x, y ):
            d = np.sqrt((p[0] - xa[:]) ** 2 + (p[1] - ya[:]) ** 2)
            zm[d <= r] = za[(d <= r)]

        ZM = zm.reshape( np.shape( zz ) )

    if ( xyr is not None ) and ( convexhull is True ) :
       mhull = xy_in_hull( xa, ya, ( x, y ), buffer=r, plot=False )[0]  
       Mhull = mhull.reshape( np.shape( zz ) ) 
       ZM = np.copy( zz )
       ZM[ ~Mhull ] = np.nan
        
    if xyr is None:
        ZM = zz

    if mask is not None:
        ZM[ np.isnan( mask ) ] = np.nan
        
    isfinit = np.isfinite( ZM )
    
    if plot == True:
        plta(zz, vmin, vmax, tit='original', sbplt=[1, 2, 1])
        plta(ZM, vmin, vmax, tit='mask', sbplt=[1, 2, 2])
        
    return ZM, isfinit

#------------------------------------------------------------------------------
def resampling( array, factor, spl_order=1, dimention='2D', mode='nearest', 
                plot=False, vmin=None, vmax=None, cval=0.0, nan=True ) :

    if type( array ) in ( list, tuple ) :
        
        if len( array ) == 3 :
            ax, ay, az = array[0], array[1], array[2]
        if len( array ) == 2 :  
            ax = np.copy( array )            
        if len( array ) == 1 :  
            az = np.copy( array )
        IsXy = True    

    else :
        az = np.copy( array )
        ax, ay = None, None
        IsXy = False
        
    if type( factor ) in ( list, tuple ) :
        factor = factor[0]/array.shape[0], factor[1]/array.shape[1]
        
    if ( np.isnan( az ).any() ) and ( spl_order > 1 ) :
        inan = np.isnan( az )
        az = fillnan( az, method='nearest' )
        IsNan = True
    else :
        IsNan = False
        
    azr = sp.ndimage.zoom( az, factor, order=spl_order, mode=mode, prefilter=True ) 
    
    if IsNan == True :
        az [ inan ] = np.nan
        inanr = sp.ndimage.zoom( inan*1, factor, order=0, mode=mode, prefilter=True ).astype(bool)
        azr[ inanr ] = np.nan
    
    if plot == True :
        
        if dimention == '2D' :
            plta( az, sbplt=[1,2,1], tit='Original', vmin=vmin, vmax=vmax )
            plta( azr, sbplt=[1,2,2], tit='Resampled', vmin=vmin, vmax=vmax )
            
        if dimention == '1D' :
            plt.figure()
            plt.plot( az, c='k', label='Original' )
            plt.plot( azr, c='b', label='Resampled' )
            
    if IsXy == True :    
        axr = sp.ndimage.zoom( ax, factor, order=1 )  
        ayr = sp.ndimage.zoom( ay, factor, order=1 ) 
        
        if nan == False :
            axr = axr[ ~inanr ]
            ayr = axr[ ~inanr ]
            azr = axr[ ~inanr ]
        
        return [ axr, ayr, azr ]
    
    else :
        
        return azr
    
# -----------------------------------------------------------------------------
def neighbors_points( xy1, xy2, radius, method='circle', plot=False, s=2 ) :
    
    x1_i, y1_i = xy1 
    shape_i = x1_i.shape
    x2, y2 = xy2
    lim2 = [ np.min(x2)-radius*2, np.max(x2)+radius*2, 
             np.min(y2)-radius*2, np.max(y2)+radius*2 ] 
    
    x1, y1, idx_i = xy_in_lim( x1_i, y1_i, lim2 )    
    s1, s2 = x1.size, x2.size
#    shape = x1.shape
    x1, y1 = x1.ravel(), y1.ravel()
    x2, y2 = x2.ravel(), y2.ravel()
    
    if method == 'circle' :
        
        x1, y1 = x1.reshape( (s1,1) ), y1.reshape( (s1,1) )
        x2, y2 = x2.reshape( (1,s2) ), y2.reshape( (1,s2) )
        X1 = np.repeat( x1, s2, 1 )
        Y1 = np.repeat( y1, s2, 1 )
        X2 = np.repeat( x2, s1, 0 )
        Y2 = np.repeat( y2, s1, 0 )   
        
        R2 = (X2 - X1)**2 + (Y2 - Y1)**2
        
        idx_f = np.sum( R2 < radius**2, axis=1 ) > 0
#        idx_f = idx_f.reshape( shape )
        idx_i[ idx_i ] = idx_f
        idx = idx_i.reshape( shape_i )
        
    x_new, y_new = x1_i[idx], y1_i[idx]
    
    if plot is True :
        
        plt.scatter( x1, y1, c='k', s=s, label='original points' )
        plt.scatter( x_new, y_new, c='r', s=s, label='selected points' )
        
    return x_new, y_new, idx

# -----------------------------------------------------------------------------
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = sp.spatial.Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = MultiLineString(edge_points)
    triangles = list( shapely.ops.polygonize(m) )
    
    return shapely.ops.cascaded_union(triangles), edge_points
    
# -----------------------------------------------------------------------------
def xy2convexhull( x, y, plot=False, close=True, size=0.5, color='b' ) :

    points = np.column_stack( ( x.ravel(), y.ravel() ) ) 
    
    hull = sp.spatial.ConvexHull( points )    
    
    xh, yh = points[hull.vertices,0], points[hull.vertices,1]
    
    if close is True :
        xh, yh = np.append( xh, xh[0] ), np.append( yh, yh[0] ) 
    
    if plot is True :
        
        plt.scatter( x, y, s=size, c=color  )
        plt.plot(xh, yh, 'r--', lw=2)
        
    return [ hull, xh, yh ]    

# -----------------------------------------------------------------------------
def xy_in_hull( x, y, hull, buffer=0, plot=False ) :

    original_shape = np.copy( x.shape )
    
    x, y = x.ravel(), y.ravel()
    
    if type( hull ) in ( list, tuple ) :
        xp, yp = hull[0], hull[1]
        hull,_,_ = xy2convexhull( xp, yp )
            
    xy = np.column_stack( ( x, y ) )    
    
    in_hull = np.all( np.add(np.dot(xy, hull.equations[:,:-1].T ),
                      hull.equations[:,-1] ) <= buffer, axis=1 )
    
    xh, yh = xp[hull.vertices], yp[hull.vertices]
    xh, yh = np.append( xh, xh[0] ), np.append( yh, yh[0] ) 

    x, y = x.reshape( original_shape ), y.reshape( original_shape )
    in_hull = in_hull.reshape( original_shape ) 
    
    if plot == True :
        plt.plot( xh, yh, 'r--', lw=2 )
        plt.scatter( x[in_hull], y[in_hull] )
    
    return in_hull, x[in_hull], y[in_hull] 


# -----------------------------------------------------------------------------
def mask2hull( mask, nan=None, plot=False, close=True, size=0.5, color='b' ) :
    
    if type( mask ) in ( list, tuple ) :
        x, y, mask = mask
    else:
        x = np.arange( mask.shape[1] )
        y = np.arange( mask.shape[0] )
        x, y = np.meshgrid( x, y )
        
    if mask.dtype != 'bool' :
        if nan is not None :
            mask = mask != nan
        else :
            mask = np.isfinite( mask )

    x = x[ mask ]
    y = np.flipud(y)[ mask ]  
    
    hull_list = xy2convexhull( x, y, plot=plot, close=close, size=size, color=color )
    
    return hull_list
    
# -----------------------------------------------------------------------------
def filt_voids( xyzgrid, xyr, step=None, plot=False, vmin=None, vmax=None, 
                method='mean' ) :  
    
    _, isf = mask2D( xyr, xyzgrid, convexhull=True ) 
#    _, isf2 = mask2D( xyr, xyzgrid ) 
    array = np.copy( xyzgrid[2] )
    m = np.argwhere( isf )
#    plta( m*1.0, lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]) )
    
    xx, yy, zz = xyzgrid
    xp, yp, r = np.copy( xyr )
    x, y = np.concatenate( ( xp, xx[~isf] ) ), np.concatenate( ( yp, yy[~isf] ) )
    
    if method == 'mean' :
        func = np.nanmean
    if method == 'median' :
        func = np.nanmedian
        
    if step is None : 
        step = min_dist( x, y )['mean']
    
    for i,j in m :
        isin = np.full( x.shape, False )
        inc = 0
        while  np.sum( isin ) < 3 :
            radius = r + step * inc    
            isin = ( x > xx[i,j] - radius ) & ( x < xx[i,j] + radius ) & \
                   ( y > yy[i,j] - radius ) & ( y < yy[i,j] + radius )
            inc += 1
            
        if inc == 0 : 
            continue
        
        win = ( xx > xx[i,j] - radius ) & ( xx < xx[i,j] + radius ) & \
              ( yy > yy[i,j] - radius ) & ( yy < yy[i,j] + radius )
    
        
        array[ i,j ] = func( array[ win ] )
        
    
    if plot == True :
        plta( xyzgrid[2], lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]), 
              vmin=vmin, vmax=vmax, tit='Original', sbplt=[1,3,1] ) 
        plta( array, lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]), 
              vmin=vmin, vmax=vmax, tit='Filtered', sbplt=[1,3,2] )
        plta( xyzgrid[2]-array, lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]), 
              vmin=None, vmax=None, tit='Differences', sbplt=[1,3,3] )        
        
    return array

# -----------------------------------------------------------------------------
def sort_points( x, y, plot=False ) :
    
    xy = np.concatenate( ( x.reshape(-1,1), y.reshape(-1,1) ), axis=1 )     

    # make PCA object
    pca = PCA(2)
    
    # fit on data
    pca.fit(xy)
    
    #transform into pca space   
    xypca = pca.transform(xy) 
    newx = xypca[:,0]
    newy = xypca[:,1]

    #sort
    indexSort = np.argsort(x)
    newx = newx[ indexSort ]
    newy = newy[ indexSort ]

    #add some more points (optional)
#    f = sp.interpolate.interp1d(newx, newy, kind='linear')        
#    newx = np.linspace(np.min(newx), np.max(newx), 100)
#    newy = f( newx )            

    #smooth with a filter (optional)
#    window = 5
#    newy = sp.signal.savgol_filter(newy, window, 2)

    #return back to old coordinates
    xyclean = pca.inverse_transform( np.concatenate( ( newx.reshape( -1, 1 ), 
                                     newy.reshape( -1, 1 ) ), axis=1) )
    x_new = xyclean[ :, 0 ]
    y_new = xyclean[ :, 1 ]

    if plot == True :
        plt.plot( x_new, y_new )
        plt.scatter( x_new, y_new )
        
    return indexSort, x_new, y_new

# -----------------------------------------------------------------------------
def mosaic_array( array_list, plot=False, vmin=None, vmax=None, ref_shape=0,
                 spl_order=1) :
    
    shape = array_list[ ref_shape ].shape
    
    array_mos = np.copy( array_list[ ref_shape ] )
    
    new_list = []
    
    for i, a in enumerate( array_list ) :
        shape_ratio = shape[0]/a.shape[0], shape[1]/a.shape[1]
        ar = sp.ndimage.zoom( a, shape_ratio, order=spl_order )
        new_list.append( ar )
        isn = np.isnan( array_mos ) 
        array_mos[isn] = ar[isn]
        
    if plot == True :
        plta( array_mos, vmin=vmin, vmax=vmax)    
        
    return array_mos, new_list

# -----------------------------------------------------------------------------
def del_in_lim( x, y, array, lim, remove='in' ) :
    
    ar = np.copy( array )
    idx = xy_in_lim( x, y, lim )[2]
    
    if remove == 'in' :
        ar = ar[ ~idx ] 
        x = x[ ~idx ]
        y = y[ ~idx ]
    
    if remove == 'out' :
        ar = ar[ idx ]    
        x = x[ idx ]
        y = y[ idx ]        
        
    return [x,y,ar]
    
# -----------------------------------------------------------------------------
def XY2bbox( X, Y, sx=0, sy=None ) :
    
    if sy is None : 
        sy = sx
    
    minbbX = np.nanmin( X ) - sx/2
    maxbbX = np.nanmax( X ) + sx/2
    
    minbbY = np.nanmin( Y ) - sy/2
    maxbbY = np.nanmax( Y ) + sy/2    
    
    bbox = [ minbbX, maxbbX, minbbY, maxbbY ]
    
    return bbox

# -----------------------------------------------------------------------------
def julian2date( jday, year, array=True, array_type=dict ) :
    
    if ( type(jday) is int ) and ( type(year) is int ) :
        date = datetime.datetime(int(year), 1, 1)+datetime.timedelta(days=int(jday) -1) 
        
    else :
        date = []
        for y, d in zip( year, jday ) :
            date.append( datetime.datetime(int(round(y)), 1, 1)+\
                         datetime.timedelta(days=int(round(d)) -1) )
            
        if array is True :
            if array_type == 'datetime64[D]' :
                date = np.array( date, dtype='datetime64[D]' )
            if array_type == str :
                date = np.array( date, dtype='datetime64[D]' )
                date = date.astype( str )
            if array_type == dict :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )
                date = { 'year':date[:,0], 'month':date[:,1], 'day':date[:,2] }
            if array_type == 'array' :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )

    return date

# -----------------------------------------------------------------------------
def date2julian( yy, mm, dd ) :
    
    if type( yy ) in ( float, int ) :
        yy = np.array( [ yy ] )
        mm = np.array( [ mm ] )
        dd = np.array( [ dd ] )
        
    jday = np.zeros( yy.shape )    
    for i,_ in enumerate( yy ):
        
        day = datetime.date( yy[i], mm[i], dd[i] ) 
        day = day.toordinal() + 1721424.5
        jday[i] = day
    
    return jday

# -----------------------------------------------------------------------------
def date2datetime( date, time, fmt='%d-%m-%y %H:%M:%S', array=True,
                   array_type='datetime64[ms]' ) :
    
    if ( type( date ) is str ) and ( type( time ) is str ) :
        date_time_str = date + ' ' + time 
        date_time = datetime.strptime( date_time_str, fmt )
        
    else :   
        date_time = []
        for d, t in zip( date, time ) :
            date_time_str = d + ' ' + t 
            date_time.append( datetime.datetime.strptime( date_time_str, fmt ) ) 
            
        if array is True :
            # a = pd.to_datetime( date_time )
            np_date_time = np.array( date_time, dtype=array_type )   
            
        if array is True :
            if array_type == 'datetime64[D]' :
                date = np.array( date, dtype='datetime64[D]' )
            if array_type == str :
                date = np.array( date, dtype='datetime64[D]' )
                date = date.astype( str )
            if array_type == dict :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )
                date = { 'year':date[:,0], 'month':date[:,1], 'day':date[:,2] }
            if array_type == 'array' :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )
    
    return np_date_time

# -----------------------------------------------------------------------------
def print_table( table, headers=None, space=12, decimals=2, rows=[], cols=[],
                 exclude=[] ):
    
    if type( table ) == dict :
        if exclude != [] :
            table = copy.copy( table )
            
            for k in exclude :
                del table[ k ]
        table, headers = dict2array( table )
        
    if type( table ) in ( list, tuple ) :
        for i, t in enumerate( table ) :
            if type( t ) in ( list, tuple ) :
                ti = np.array( t )
            if i == 0 :
                array = ti
            else :
                array = np.column_stack( ( array, ti ) )
        table = array.copy()        
        
    if type( decimals ) in ( int, float ) :
        decimals = [ decimals ] * len(table[ 0 ] )


    is_int = ( table % 1 ) == 0
    
    # Do heading
    if headers is None : 
        print("     ", end="")
        for j in range(len(table[ 0 ] ) ):
            print(f"% {space}d" % j, end="")
        print()
        print("     ", end="")
        for j in range(len(table[0])):
            print("-"*space, end="")
        print()
        
    else :
        print("     ", end="")
        for j in headers:
            print(f"% {space}s" % j, end="")
        print()
        print("     ", end="")
        for j in range( len( table[0] ) ):
            print("-"*space, end="")
        print()        
        
    if type( rows ) == int :
        rows = list( range( rows ) )           
        
    if type( cols ) == int :
        cols = list( range( cols ) )  
        
    if ( rows == [] ) or ( rows is None ) :
        rows = list( range( len( table[:,0] ) ) )          
        
    if ( cols == [] ) or ( cols is None ) :
        cols = list( range( len( table[0] ) ) )  
        
    # Matrix contents
    for i in range(len(table)):
        if i not in rows : continue
        print("%3d |" % (i), end="") # Row nums
        for j in range( len( table[0] ) ) :
            if j not in cols : continue
            if is_int[i,j] == True :
                ft = 'd'
            else :
                ft = f'.{decimals[j]}f'
            print(f"% {space}{ft}" % (table[i][j]), end="")
        print()  
        
   # for i in range(len(table)):
   #      if i not in rows : continue
   #      print("%3d |" % (i), end="") # Row nums
   #      for j in range( len( table[0] ) ) :
   #          if j not in cols : continue
   #          if (isinstance( table[i,j], int )) or ((isinstance( table[i,j], float )) and ( ( table[i,j] % 1 ) == 0 ) ) :
   #          # if is_int[i,j] == True :
   #              ft = 'd'
   #              print(f"% {space}{ft}" % (table[i][j]), end="")
   #          if (isinstance( table[i,j], float )) and ( ( table[i,j] % 1 ) != 0 ) :
   #              ft = f'.{decimals[j]}f'
   #              print(f"% {space}{ft}" % (table[i][j]), end="")
   #          if (isinstance( table[i,j], str )) :
   #              print(f"{table[i][j]: >{space}}", end="")
    
    return table

# -----------------------------------------------------------------------------
def combine64( years, months=1, days=1, weeks=None, hours=None, minutes=None,
               seconds=None, milliseconds=None, microseconds=None, nanoseconds=None ):
    
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1

    types = ( '<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
              '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]' )
    
    if np.any( np.mod( seconds, 1 ) != 0 ) :
        nanoseconds = seconds.copy() * 1e9
        seconds=None
    
    vals = ( years, months, days, weeks, hours, minutes, seconds,
             milliseconds, microseconds, nanoseconds )
    
    datetime_type = np.sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
                    if v is not None)
    
    return datetime_type

# -----------------------------------------------------------------------------
def read_file( file, rows=None ) :
    
    f = open( file, 'r' )
    
    lines = [line.rstrip() for line in f]
    
    if type( rows ) == int :
        rows = list( range( rows ) )           
        
    if ( rows == [] ) or ( rows is None ) :
        rows = list( range( len( lines ) ) )   
        
    lines = [ lines[index] for index in rows ]
    
    f.close()
    
    return lines

# -----------------------------------------------------------------------------
def open_dir( path ) :
    
    if os.name == 'nt' :
        os.system( 'start ' + path )
        
    if os.name == 'posix' :
        os.system( 'nautilus ' + path + '&' )
        
# -----------------------------------------------------------------------------
def del_files( path, string ) : 
    
    for file in os.listdir( path ) :
         
        if string in file :
            
            if os.path.isfile( path + os.sep + file ) :
            
                os.remove( path + os.sep + file )
                
            if os.path.isdir( path + os.sep + file ) :
            
                os.rmdir( path + os.sep + file )
            
# -----------------------------------------------------------------------------
def ls( path ) : 
    
    for file in os.listdir( path ) :
        
        print( file )    
        
# -----------------------------------------------------------------------------
def xyz2grid( x, y, z, lim=None, extend=None, extend_method='percentage', 
              sqr_area=False, gstep=None, blkm=None, method='linear', prjcode_in=4326,
              prjcode_out=4326, filt=None, filt_radius=1, msk_radius=None, msk_shp=None,
              in_out='in', fill_nan=None, plot=False, vmin=None, vmax=None,
              pltxy=False, s=None, msk_arr=None, adjst_lim=False, iter=1, tension=0.35,
              filt_factor=2, filt_sigma=1, padw=0, pmode='gdal' ):
              
   
    if prjcode_in != prjcode_out:
        x, y = prjxy( prjcode_in, prjcode_out, x, y )  
    
    if lim is None:
        lim = [np.min(x), np.max(x), np.min(y), np.max(y)]
        xl, yl, zl = x, y, z
    else:
        xl, yl, indx = xy_in_lim( x, y, lim, extend=33 )
        zl = z[indx]
        
    if extend is not None:
        lim = extend_lim(lim, extend, extend_method, sqr_area )

    if gstep == None:
        gstep = min_dist( xl, yl )['mean']
    
    if blkm != None :    
        if blkm == True:
            xl, yl, zl = block_m( xl, yl, zl, gstep, lim=lim )
        if type( blkm ) in ( int, float ) :
            xl, yl, zl = block_m( xl, yl, zl, blkm, lim=lim )
    
    if adjst_lim is False :    
        xg = np.arange( lim[0], lim[1], gstep )
        yg = np.arange( lim[3], lim[2], -gstep )
    if adjst_lim is True :   
        xg = np.linspace( lim[0], lim[1], int( ( lim[1] - lim[0] ) / gstep ) )
        yg = np.linspace( lim[3], lim[2], int( ( lim[3] - lim[2] ) / gstep ) )
    xx, yy = np.meshgrid(xg, yg)
       
#    points = np.column_stack((xl, yl))
    if method == 'surface':
        xx, yy, zz = gmt_surf( xl, yl, zl, lim=lim, gstep=gstep, tension_factor=tension )
    else:
#        zz = scy.interpolate.griddata(points, zl, (xx, yy), method=method)
        zz = xyz2xy( ( xl, yl, zl ), (xx, yy), method=method, fillnan=False )[0]

    if fill_nan is not None :
        if ( type(fill_nan) == bool ) and ( fill_nan is True ) :
            fill_nan = 'gdal'
        zz = fillnan( zz, xy=( xx, yy ), method=fill_nan, iter=iter ) 
        
    if filt is not None :
        zz = filt2d( zz, ftype=filt, iter=iter, radius=filt_radius, 
                         factor=filt_factor, sigma=filt_sigma, padw=padw, pmode=pmode )        
    
    if msk_radius is not None:
        zz = mask2D( ( xl, yl, msk_radius ), (xx, yy, zz) )[0]
        
    # if msk_shp is not None:
    #     zz = rt.mask_array( xx, yy, zz, msk_shp, prjcode=prjcode_out )

    if msk_arr is not None:
        zz[np.isnan( msk_arr ) ] = np.nan         

    if plot == True:
        if pltxy == False:
            plta( zz, vmin=vmin, vmax=vmax, cmap='rainbow' )
        if pltxy == True:
            limp = xy2lim( xx, yy ) 
            plta( zz, vmin=vmin, vmax=vmax, cmap='rainbow', lim=limp, points=[ xl, yl ] )
            
    return [ xx, yy, zz ], [ x, y, z ]
    
# -----------------------------------------------------------------------------
def relative_max_min_2D( array, n=2 ) :
    
    ar = np.copy( array ) 
    MD = np.zeros(ar.shape)
    
    xu = np.arange( 0, ar.shape[1] )
    yu = np.arange( 0, ar.shape[0] )
    
    xx, yy = np.meshgrid( xu, yu )    

    # -------------------------------------------------------------------------
    # Edge points generation 
    
    for i, j in np.ndindex(ar.shape):
        
        if i==0 or i==ar.shape[0]-1 or \
           j==0 or j==ar.shape[1]-1:
            continue
        
        Np=0
        Nn=0
        
        if ar[i-1, j-1] < ar[i, j] > ar[i+1, j+1]:
            Np +=1
        if ar[i-1, j] < ar[i, j] > ar[i+1, j]:
            Np +=1
        if ar[i-1, j+1] < ar[i, j] > ar[i+1, j-1]:
            Np +=1
        if ar[i, j-1] < ar[i, j] > ar[i, j+1]:
            Np +=1
        
        if ar[i-1, j-1] > ar[i, j] < ar[i+1, j+1]:
            Nn +=1
        if ar[i-1, j] > ar[i, j] < ar[i+1, j]:
            Nn +=1
        if ar[i-1, j+1] > ar[i, j] < ar[i+1, j-1]:
            Nn +=1
        if ar[i, j-1] > ar[i, j] < ar[i, j+1]:
            Nn +=1            
            
        if Np > n :
            MD[i,j] = 1
        if Nn > n :
            MD[i,j] = -1
            
    xmin, ymin = xx[ MD==-1 ], yy[ MD==-1 ]
    xmax, ymax = xx[ MD==1 ], yy[ MD==1 ]
    ax = plta( ar )
    ax.scatter( xmin, ymin )
    ax.scatter( xmax, ymax )
        
    return MD


# -----------------------------------------------------------------------------
def rotate_xy( x, y, radians, shift=True, fix_point='first', plot=False ) :

    if shift is not None :
        
        if shift in ( tuple, list ) :    
            x = x - shift[0]
            y = y - shift[1]
            
        if shift is True :
            xu = np.unique( x )
            idx = x == xu[0]
            
            if type( fix_point ) == str :
                
                if fix_point == 'mean' : 
                    shift = np.mean(x), np.mean(y)
                    
                if fix_point == 'first' :
                    shift = x[ idx ][0], y[ idx ][0]
                    
            if type( fix_point ) in ( tuple, list ) :
                shift = fix_point

            x = x - shift[0]
            y = y - shift[1]
    
    else :
        
        shift = 0, 0
    
    fix_point = shift
    
    xr = x * np.cos( radians ) + y * np.sin( radians ) + shift[0]
    yr = -x * np.sin( radians ) + y * np.cos( radians ) + shift[1]
    
    if plot is True :
        
        plt.scatter( x, y, c='b' )
        plt.scatter( xr, yr, c='b' )

    return xr, yr, fix_point

# -----------------------------------------------------------------------------
def grid_rot_angle( x, y, plot=False, treshold=1e-5, fix_point='first' ) :
    
    xu = np.unique( x )
    
    idx0 = x == xu[0]
    idx1 = x == xu[1]
    
    x0 = x[ idx0 ][0]
    y0 = y[ idx0 ][0]
    
    x1 = x[ idx1 ][0]
    y1 = y[ idx1 ][0]
    
    angle = np.abs( np.arctan( ( y1-y0 ) / ( x1-x0 ) ) )
    
    xr, yr, fix_point = rotate_xy( x, y, angle, fix_point=fix_point )
    
    diffx = np.abs( xr[ idx0 ][0] - xr[ idx1 ][0] )
    diffy = np.abs( yr[ idx0 ][0] - yr[ idx1 ][0] )
    
    if ( diffx > treshold ) and ( diffy > treshold ) :
    
        xr, yr, fix_point = rotate_xy( x, y, (np.pi/2)-angle, fix_point=fix_point )   
        
        diffx = np.abs( xr[ idx0 ][0] - xr[ idx1 ][0] )
        diffy = np.abs( yr[ idx0 ][0] - yr[ idx1 ][0] )
        
        if ( diffx > treshold ) and ( diffy > treshold ) :
            
            print( "SOMETHING IS WRONG WITH THE POINTS ROTATION!!!")
            
            plot = True 
            
    decimals = int( np.abs( np.floor( np.log10( treshold ) ) ) ) - 1
    
    xr = np.round( xr, decimals )
    yr = np.round( yr, decimals )
    
    xru = np.unique( xr )
    yru = np.unique( yr )

    Xr, Yr = np.meshgrid( xru, yru )       
    xr = Xr.ravel()
    yr = Yr.ravel()
    
    if plot == True :

        plt.scatter( x.ravel(), y.ravel(), c='b', label='original' )        
        plt.scatter( xr.ravel(), yr.ravel(), c='r', label='rotated' )
        plt.legend()
        
    return angle, xr, yr, fix_point
    
# -----------------------------------------------------------------------------
def mesh3Dmodel( x=None, y=None, z=None, lim=None, step=None, plot=False, 
                 sbplt=111, out_type='dictionary', size=None, centre_xy=False, 
                 xm=None, ym=None, treshold=1e-5, fix_point='first' ) :

    if step is None :
        x, y = np.copy( x ), np.copy( y )
        angle, x, y, fix_point = grid_rot_angle( x, y, treshold=treshold, fix_point=fix_point ) 
        xu = np.unique( x )
        yu = np.unique( y )
        zu = np.unique( z )

    if type(step) in (int, float) :
        step = step, step, step             

    if ( lim is not None ) and ( step is not None ) and ( centre_xy is True ) :
        if xm is None :
            xm = np.round( np.mean( ( lim[1], lim[0] ) ) )
        xr = np.max( (int((xm-lim[0])/step[0]), int((lim[1]-xm)/step[0]) ) )
        lim[0] = xm - step[0] * xr
        lim[1] = xm + step[0] * xr
        if ym is None :
            ym = np.round( np.mean( ( lim[2], lim[3] ) ) )
        yr = np.max( (int((ym-lim[2])/step[1]), int((lim[3]-ym)/step[1]) ) )
        lim[2] = ym - step[1] * yr
        lim[3] = ym + step[1] * yr

    if x is not None :
        xu = np.unique( x )
    else :
        xu = np.arange( lim[0], lim[1]+step[0], step[0] )

    if y is not None :
        yu = np.unique( y )
    else :
        yu = np.arange( lim[2], lim[3]+step[1], step[1] )

    if z is not None :
        zu = np.unique( z )
    else :
        zu = np.arange( lim[5], lim[4]-step[2], -step[2] )

    if lim is not None :
        xu = xu[ (xu>=lim[0]) & (xu<=lim[1]) ]
        yu = yu[ (yu>=lim[2]) & (yu<=lim[3]) ]
        zu = zu[ (zu>=lim[4]) & (zu<=lim[5]) ]

    X, Y, Z = np.meshgrid( xu, yu, zu )
    shape = X.shape

    if step is None :
        angle, X, Y, fix_point = grid_rot_angle( X, Y, treshold=treshold, fix_point=fix_point ) 

    if plot is True :
        if size is None :
            size = 1
        ax = plt.subplot(sbplt, projection='3d')
        ax.scatter( X[:,int(shape[1]/2),:].ravel(), 
                    Y[:,int(shape[1]/2),:].ravel(), 
                    Z[:,int(shape[1]/2),:].ravel(), s=size, c='b' )
        ax.scatter( X[int(shape[0]/2),:,:].ravel(), 
                    Y[int(shape[0]/2),:,:].ravel(), 
                    Z[int(shape[0]/2),:,:].ravel(), s=size, c='b' )

    if out_type == '3darray' :
        return X, Y, Z
    if out_type == '1darray' :
        return X.ravel(), Y.ravel(), Z.ravel(), shape
    if out_type == 'dictionary' :
        return { 'x':X.ravel(), 'y':Y.ravel(), 'z':Z.ravel() }, shape    

# -----------------------------------------------------------------------------
def despike_1D( array, treshold, radius=2, pmode='median', plot=False, kind='cubic' ) :

    signal = np.copy( array )

    psignal = np.pad( signal, radius, pmode)

    inan = np.full( psignal.shape, False ) 
    for i, e in enumerate( psignal ) :
        if ( i < radius ) or ( i > len( signal )-1 ) :
            continue

        meanw = np.mean( psignal[i-radius:i+radius] )
        if ( e > meanw + treshold ) or ( e < meanw - treshold ) :
            inan[ i ] = True 

    x = np.arange(1, len(psignal)+1, 1)
    print( x.shape, psignal.shape )
    f = sp.interpolate.interp1d( x[~inan], psignal[~inan], kind=kind )

    psignal[ inan ] = f( x[ inan ] )  

    new_signal = psignal[ radius : -radius ]

    if plot == True :

        plt.plot( signal, c='b', label='original' )
        plt.plot( new_signal, c='r', label='despiked' )
        plt.legend()

    return new_signal 


# -----------------------------------------------------------------------------
def xyzp2array3d( x, y, z, p, xi=None, yi=None, zi=None, lim=None, step=None, plot=False, 
                  sbplt=111, size=None, centre_xy=False, method='nearest' ) :
    
    """
    Convert coordinates (x, y, z) and a function (p) into a 3D numpy array.
    
    Args:
        x (array-like): x-coordinates.
        y (array-like): y-coordinates.
        z (array-like): z-coordinates.
        p (function or list or tuple): Function or list/tuple of functions.
        xi (array-like, optional): x-coordinates for the grid. Defaults to None.
        yi (array-like, optional): y-coordinates for the grid. Defaults to None.
        zi (array-like, optional): z-coordinates for the grid. Defaults to None.
        lim (array-like, optional): Limits of the grid. Defaults to None.
        step (float, optional): Step size for the grid. Defaults to None.
        plot (bool, optional): Whether to plot the 3D array. Defaults to False.
        sbplt (int, optional): Subplot specification for the plot. Defaults to 111.
        size (float or tuple, optional): Size of the plot. Defaults to None.
        centre_xy (bool, optional): Whether to center the x and y coordinates. Defaults to False.
        xm (float, optional): Midpoint of x. Defaults to None.
        ym (float, optional): Midpoint of y. Defaults to None.
        method (str, optional): Interpolation method. Defaults to 'nearest'.
    
    Returns:
        tuple: 3D numpy array (X, Y, Z) and the interpolated values (P).
    """
                  
    if xi is None :
        xi = x
    if yi is None :
        yi = y
    if zi is None :
        zi = z

    X, Y, Z = mesh3Dmodel( x=xi, y=yi, z=zi, lim=lim, step=step, out_type='3darray', 
                 centre_xy=centre_xy )
    
    if type(p) not in ( list, tuple ) :
        p = [ p ]

    P = []

    for i in p :
        Pi = sp.interpolate.griddata( ( x, y, z), i ( X, Y, Z ), method=method ) 
        P.append( Pi )

    if len( P ) == 1 :
        P = P[ 0 ]

    shape = X.shape

    if plot is True :
        if size is None :
            size = 1
        ax = plt.subplot(sbplt, projection='3d')
        ax.scatter( X[:,int(shape[1]/2),:].ravel(), 
                    Y[:,int(shape[1]/2),:].ravel(), 
                    Z[:,int(shape[1]/2),:].ravel(), s=size, c=P[ 0 ][:,int(shape[1]/2),:].ravel() )
        ax.scatter( X[int(shape[0]/2),:,:].ravel(), 
                    Y[int(shape[0]/2),:,:].ravel(), 
                    Z[int(shape[0]/2),:,:].ravel(), s=size, c=P[ 0 ][int(shape[0]/2),:,:].ravel() )
        
    if len( P ) == 1 :
        P = P[ 0 ]

    return X, Y, Z, P

# -----------------------------------------------------------------------------
def write_xyzp2vtk( x, y, z, p, path, name, nan2num=0 ):
    
    file_vtk = name + '.vtk'
    os.makedirs(path, exist_ok=True)  
    pf_new = os.path.join(path, file_vtk)

    idx = np.lexsort( ( x,y,z ) ) 
    p = p[ idx ]

    x_coordinates = np.unique( x )
    y_coordinates = np.unique( y )
    z_coordinates = np.unique( z )

    x_dim = np.size( x_coordinates )
    y_dim = np.size( y_coordinates )
    z_dim = np.size( z_coordinates )

    if nan2num is not None :
        inan = np.isnan( p )
        p[ inan ] = nan2num

    with open(pf_new, 'w') as fid:
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('Gravity-Model %s\n' % name)
        fid.write('ASCII\n')
        fid.write('DATASET RECTILINEAR_GRID\n')
        fid.write('DIMENSIONS %d\t%d\t%d\n' % (x_dim, y_dim, z_dim))
        fid.write('X_COORDINATES %d double\n' % x_dim)
        fid.write(' '.join(map(str, x_coordinates)) + '\n')
        fid.write('Y_COORDINATES %d double\n' % y_dim)
        fid.write(' '.join(map(str, y_coordinates)) + '\n')
        fid.write('Z_COORDINATES %d double\n' % z_dim)
        fid.write(' '.join(map(str, z_coordinates)) + '\n')
        fid.write('POINT_DATA %d\n' % (x_dim * y_dim * z_dim))
        fid.write('SCALARS Density double\n')
        fid.write('LOOKUP_TABLE default\n')
        for ind_p in range( len( p ) ):
            fid.write('%f \n' % p[ind_p] )

    fid.close()  
     # File is automatically closed after the 'with' block