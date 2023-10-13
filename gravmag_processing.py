# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:55:41 2019

@author: lzampa
"""

import os
mdir = os.path.dirname( os.path.abspath(__file__) ) 

import numpy as np
import os
from collections import namedtuple
from datetime import datetime, timedelta
import lszpy.plot as lz_plot
import matplotlib.pyplot as plt
import harmonica as hm
import datetime as dt
import copy
import lszpy.utils as utl
import lszpy.te_harmonica as te_hm

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
def g_shpere( dist=0, radius=0, density=None, mass=None ) :
    
    if mass is None :
        
        volume = ( 4/3 )  * np.pi * radius**3
        mass = density * volume
        
    g_sph = G * mass / ( ( dist + radius )**2 )

    return g_sph
    
# -----------------------------------------------------------------------------
def gn_67(lat):
    """
    Normal gravity calculated with the International Gravity Formula 1967 [mGal].
    Based on Hayford Elipsoid     
    """  
    
    a = 978031.846
    b = 0.005278895
    c = 0.000023462
    gn1 = b*np.power(np.sin(np.deg2rad(lat)),2)
    gn2 = +c*np.power(np.sin(np.deg2rad(lat)),4)
    gn3 = a*(1+gn1+gn2)
    
    return gn3

# -----------------------------------------------------------------------------
def gn_80(lat):
    """
    Normal gravity calculated with the International Gravity Formula GRS80 [mGal]
    
    Ref:
    - Hinze et all., 2005 doi:10.1190/1.1988183
    """ 
          
    a = 978032.67715 # ge
    b = 0.001931851353 # k
    c = 0.0066943800229 # e^2  
    gna = 1+b*np.power(np.sin(np.deg2rad(lat)),2) # numerator
    gnb = np.sqrt(1-c*np.power(np.sin(np.deg2rad(lat)),2))  # denominator
    gnc = a*gna/gnb
    return gnc

# -----------------------------------------------------------------------------
def gn_84(lat):
    """
    Normal gravity calculated with WGS84 gravity formula [mGal]    
      
    Ref:
    - Decker, B. L. (1986). World geodetic system 1984
    """ 
    
    a = 978032.53359 
    b = 0.00193185138639
    c = 0.00669437999013
    lat = np.deg2rad(lat)
    gna = 1+b*np.power(np.sin(lat),2)
    gnb = np.sqrt(1-c*np.power(np.sin(lat),2))
    gnc = a*gna/gnb
    
    return gnc
    
# -----------------------------------------------------------------------------
def atm_c(h):
    """
    Atmospheric correction [mGal] 
    Ref:
    - Hinze et all., 2005 doi:10.1190/1.1988183
    """  

    a = 0.874
    b = 9.9*(10**(-5))
    c = 3.56*(10**(-9))
    gnb = b*h
    gnc = c*np.power(h,2)
    
    return a-gnb+gnc

# -----------------------------------------------------------------------------
def fa_c(h, lat=0, model='ell', R=R_wgs84):
    """
    Free Air Correction
    
    model=='sph': spherical approximation, with mean radius R [m]
    model=='ell': ellipsoidal approximation
    
    Ref:
    - Hinze et all., 2005 doi:10.1190/1.1988183
    """ 
    
    if model=='sph':
        fac = (-2*G*M/(R_wgs84**3))*h*1e5  
    if model=='ell':
        a = 0.3087691
        b = 0.0004398
        c = 7.2125*10**(-8) 
        fac = -(a-b*np.power(np.sin(np.deg2rad(lat)),2))*h+c*(np.power(h,2))
    
    return fac

# -----------------------------------------------------------------------------
def slab(h, dc=2670, dw=1030, topo_sea=False, st_type=None ):
    """
    Gravity effect of a flat slab [mGal]
    h = thickness    
    d = density    
    """
    
    if st_type is not None :
        ms = st_type == 1
        g_slb = 2 * np.pi * G * dc * h * 1e5
        g_slb[ms] = 2 * np.pi * G * ( -dw + dc ) * h[ms] * 1e5        
    
    if ( topo_sea is True ) and ( st_type is None ) :
        ms = h < 0
        g_slb = 2 * np.pi * G * dc * h * 1e5
        g_slb[ms] = 2 * np.pi * G * ( -dw + dc ) * h[ms] * 1e5
        
    else:
        g_slb = 2 * np.pi * G * dc * h * 1e5
    
    return g_slb

# -----------------------------------------------------------------------------
def sph_shell(h, d):
    """
    Gravity effect of a spherical shell [mGal]
    h = thickness    
    d = density  
    """
    
    g_sphc = 4*np.pi*G*d*h*1e5   
    
    return g_sphc

# -----------------------------------------------------------------------------
def curv_c(h=1, Rt=R_wgs84, Rd=167000, dc=2670, dw=1030, units=1e5, st_type=None):
    """
    Curvature correction for slab approx.
    
    h = station height [m]
    Rt = mean earth radius [m]
    Rd = slab radius [m]
    dc = crust density [kg/m3]
    dw = water density [kg/m3]
    units = if 1=[m/s2], if 1e5=[mGal]
    
    Ref:
    - Fullea et al., 2008 - FA2BOUG—A FORTRAN 90 code to compute Bouguer gravity anomalies 
          from gridded free-air anomalies: Application to the Atlantic-Mediterranean transition zone;    
    - Whitman, 1991 - A microgal approximation for the Bullard B—earth’s curvature—gravity correction;    
    """
    
    if type(h) in (int, float) : 
        h=[h]
    alph = Rd/Rt
    cc = np.zeros(np.size(h))
    for i in range(0,np.size(h)):
        eta = h[i]/(Rt+h[i])
        
        if st_type == None :    
            if h[i] >= 0:  
                cc[i]=2*np.pi*G*dc*h[i]*(alph/2-eta/(2*alph)-eta)*units
            if h[i] < 0:  
                cc[i]=2*np.pi*G*(dc-dw)*h[i]*(alph/2+eta/(2*alph)+eta)*units 
                
        if st_type == 0 :
            cc[i]=2*np.pi*G*dc*h[i]*(alph/2-eta/(2*alph)-eta)*units 
            
        if st_type in ( 1, 2 ) :    
            cc[i]=2*np.pi*G*(dc-dw)*h[i]*(alph/2+eta/(2*alph)+eta)*units 
            
    return cc

# -----------------------------------------------------------------------------
def fw_c(h, lat=0, g0=None, dw=1030, R=R_wgs84, a0=a_wgs84, c0=c_wgs84, 
         J2=J2_wgs84, w=w_wgs84, model='sph'):
    """
    Free Water Correction -- used for seaflor grav data [mGal] 
    
    h = station height [m] (NB. below s.l. h must be negative)
    g0 = normal gravity 
    lat = latitude [deg]
    dw =  water density [kg/m3]
    R = mean earth radius [m]
    model='sph' --> dg/dh = -(2GM)/R + 4*pi*rho_w = spherical, 1st order taylor approx.
    model='ell' --> ellipsoidal, 2nd order taylor approx.
    
    Ref: 
    - F. D. Stacey et al., 1981, doi: 10.1103/PhysRevD.23.1683
    """
    
    if g0==None:
        g0=gn_84(lat)*1e-5 - atm_c(0)
    if model=='sph':
        fwc = (-((2*G*M)/(R**3))+4*np.pi*G*dw)*h*1e5
    if model=='ell':
        # r0 = a(1-((a-c)/a)*sin2(lat))
        # where is the geocentric latitude 
        # see also https://www.eoas.ubc.ca/~mjelline/Planetary%20class/14gravity1_2.pdf
        f = ((a0-c0)/a0)
        # geographic latitude
        lat = np.deg2rad(lat)
        # geocentric latitude
        lat = np.arctan(((c0**2)/(a0**2))*np.tan(lat))
        r0 = a0*(1-((a0-c0)/a0)*(np.sin(lat)**2)) 
        r = r0+h
        a = r*(1+(((a0**2)/(c0**2))-1)*np.sin(lat)**2)
        c = a*(1-f)
        fwc = -2*(g0/r0)*h*((1 +1.5*(h/r0)) +3*J2*(1.5*(np.sin(lat)**2)-0.5)) +3*(w**2)*h*(1-np.sin(lat)**2) \
              +4*np.pi*G*(((c/a)*(1 +2*(h/r0) +0.5*((a**2/c**2)-1)*(c**2/a**2)))*dw*h -(dw*h**2)/r0)
        fwc=fwc*1e5
        
    return fwc

# -----------------------------------------------------------------------------
def ie( N, lat=0, dc=2670, dw=1030, xy=[], st_type=0, 
        method='nearest', prjcose_in=4326 ) :
    
    """
    Indarect effect -- used for disturbace correction [mGal] 
    
    N = Geoid height ( numpy array or list/tuple( x, y, N ) )
    den = density [kg/m3]
    xy = list or tuple with x and y coordinates of computational points
    method = interpolation method of geoid coordinates onto computational points x, y
    prjcode_in = proj code of input computational points coordinates (x, y)
    prjcode_out = proj code used for intepolating the geoid onto computational points
    
    Ref: 
    - Hinze et all., 2005 doi:10.1190/1.1988183
    """    
    
    if ( type(N) in (list, tuple) ) and ( len( N ) == 3 ) :
        xg, yg, N = N[0], N[1], N[2]  
        if utl.prj_(prjcose_in) != utl.prj_(4326) :
            lon, lat =utl.prjxy( prjcose_in, 4326, xg, yg )    
        
    if xy!=[] :
        x, y = xy
        N = utl.xyz2xy( ( xg, yg, N ), (x, y), method=method )[0]
        
    if ( type( st_type ) in (int, float) ) and ( type( N ) not in (int, float) ) :
        st_type = np.full( N.shape, st_type )
    
    if type( N ) not in (int, float)  :    
        den = np.full( N.shape, dc )
        idx = st_type != 0 
        den[ idx ] = dw
    else :
        if st_type != 0 :
            den = dw
        else:
            den = dc    
    
    ie = - fa_c( N, lat=lat ) 
    dg_ie_bg = 2 * np.pi * G * den * N * 1e5
    geof_ie = ie - dg_ie_bg
    
    return [ ie, geof_ie, N ]

# -----------------------------------------------------------------------------
def line_filt( xyzl, wind_size=3, prjcode_in=4326, filter_type='median', 
               poly_order=6, prjcode_out=4326, x_c=0, y_c=1, z_c=2, line_c=3,
               extend_factor=2, dist=None, deg_to_m=False, pad_mode='linear_ramp',
               new_xy=False, plot_lines=[], x_units='', y_units='[ mGal ]',
               edge_fix=True, gauss_sigma=2, order_c=None ) :
               
    
    xyzl = np.copy( xyzl )
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = utl.prjxy( prjcode_in, prjcode_out, 
                                              xyzl[:,x_c], xyzl[:,y_c] )
    
    xyzl, idx_original = utl.sort_lines( xyzl, line_c=line_c, x_c=x_c, y_c=y_c, 
                                         add_dist=False, order_c=order_c )
    
    xyzl_new = np.copy( xyzl )
    
    lines = np.unique( xyzl[ :, line_c ] )  
        
    for i, l in enumerate( lines ) :
        
        idx = xyzl[ : , line_c ] == l
        line = xyzl[ idx ]
        line_new = np.copy( line )
        
        pad_val = np.pad( line_new[ : , z_c ], wind_size*extend_factor, 
                          mode=pad_mode, end_values=10 )

        if filter_type == 'uniform' : 
            filt_val = utl.sp.ndimage.uniform_filter( pad_val, wind_size )
            
        if filter_type == 'wiener' : 
            filt_val = utl.sp.signal.wiener( pad_val, mysize=wind_size )
            
        if filter_type == 'median' : 
            filt_val = utl.sp.signal.medfilt( pad_val, kernel_size=wind_size )
            
        if filter_type == 'savgol' : 
            filt_val = utl.sp.signal.savgol_filter( pad_val, wind_size, poly_order )  
            
        if filter_type == 'gauss' : 
            filt_val = utl.ndimage.gaussian_filter1d( pad_val, sigma=gauss_sigma )            
            
        if edge_fix == True :
            diff = filt_val - pad_val
            w = np.zeros( diff.shape )
            w[0:wind_size*extend_factor+wind_size] = 1
            w[-(wind_size*extend_factor+wind_size):] = 1
            filt_w = utl.sp.ndimage.gaussian_filter1d( w, sigma=gauss_sigma ) 
            diff_w = diff * filt_w 
            filt_val = filt_val - diff_w
        
        line_new[ : , z_c ] = filt_val[ wind_size*extend_factor : 
                                        -wind_size*extend_factor ] 
            
        xyzl_new[ idx ] = line_new
        
    if plot_lines not in ( [], None, False ) :
        
        if deg_to_m == True : deg2m = True
        else : deg2m = False
        
        if plot_lines == True :
            plot_lines = []
        z_old_c = xyzl.shape[1]
        xyzl_new = np.column_stack( ( xyzl_new, xyzl[ :, z_c ] ) )
        fl = lz_plot.plot_lines( xyzl_new, line_c=line_c, x_c=x_c, y_c=y_c, z_c=[z_old_c,z_c], 
                              deg2m=deg2m, plot_points=False, marker='+', marker_color='k',
                              s=1.5, x_units=x_units, y_units=y_units, c=['b','g' ], 
                              legend=[ 'original_line', 'filtered_line' ], lines=plot_lines ) 
        
        xyzl_new = np.delete( xyzl_new, z_old_c, 1 )
        
    else : fl = None
        
    if ( new_xy is False ) and ( prjcode_in != prjcode_out )  :
        xyzl_new[:,x_c], xyzl_new[:,y_c] = utl.prjxy( prjcode_out, prjcode_in, 
                                                      xyzl_new[:,x_c], 
                                                      xyzl_new[:,y_c] )  
        
    return xyzl_new, idx_original, fl    
        
# -----------------------------------------------------------------------------
def line_remres( xyzl, xyz_ref, wind_size=None, prjcode_in=4326, prjcode_out=4326,
                 plot_lines=False, s=1, plot_cross=False, vminc=None, vmaxc=None,
                 pad_idx=-1, plot=False, vmin=None, vmax=None, new_xy=True,
                 x_c=0, y_c=1, z_c=2, line_c=3, lim=None, units=None, ref_wl=16000,
                 wind_factor=1, xyz_w=None, x_units='', y_units='[ mGal ]',
                 adjst_lev=True, power=2, iterations=1, dist=None, spl_k=3, spl_s=0,
                 median_lev=False, radius=[], median_lines=[], order_c=None,
                 ref_res=1000, pad_dist=0, filt=None, wfilt=3 ) :
    
    line_c_origin = copy.copy( line_c )
    z_c_origin = copy.copy( z_c )
    x_c_origin = copy.copy( x_c )
    y_c_origin = copy.copy( y_c )

    xyzl = np.copy( xyzl )
    xyz_ref = np.copy( xyz_ref )
    
    if type( xyz_ref ) is tuple :
       xyz_ref = [ xyz_ref[0], xyz_ref[1], xyz_ref[2] ] 
       
    if ( xyz_w !=None ) and ( type( xyz_w ) is tuple ) :
       xyz_w = [ xyz_w[0], xyz_w[1], xyz_w[2] ]        
    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = utl.prjxy( prjcode_in, prjcode_out, 
                                              xyzl[:,x_c], xyzl[:,y_c] )
        
        xyz_ref[0], xyz_ref[1] = utl.prjxy( prjcode_in, prjcode_out, 
                                            xyz_ref[0], xyz_ref[1] )   
        
        if xyz_w !=None :
            xyz_w[0], xyz_w[1] = utl.prjxy( prjcode_in, prjcode_out, 
                                            xyz_w[0], xyz_w[1] ) 
            
    xyzl_origin = np.copy( xyzl )
        
    if units == None :
        units = utl.prj_units( prjcode_out )
    
    if wind_size is None :
        ref_res = utl.min_dist( xyz_ref[0].ravel(), xyz_ref[1].ravel() )['mean']
        if wind_factor == 1 :
            if units == 'degree' :
                wind_factor = int( ref_wl / utl.deg2m( ref_res ) )
            else :
                wind_factor = int( ref_wl / ref_res )
        wind_size = ref_res * wind_factor
    if units == 'degree' :
        print( 'window_size : ', round( utl.deg2m( wind_size ), 2 ) )
    else :   
        print( 'window_size : ', round( wind_size, 2 ) )
    
    half_ws =  wind_size / 2   
    
    if pad_dist is None :
        pad_dist = wind_size
    if pad_dist != 0 :
        xyzl, idx_origin = utl.pad_lines( xyzl, pad_dist=pad_dist, pad_idx=pad_idx, 
            x_c=x_c, y_c=y_c, z_c=z_c, line_c=line_c, order_c=order_c, radius=half_ws )
        
    if dist is not None :       
            
        xyzl = utl.resamp_lines( xyzl, dist, method='linear', order_c=order_c,
                                 line_c=line_c, x_c=x_c, y_c=y_c, z_c=z_c )
        line_c = 3
        z_c = 2
        x_c = 0
        y_c = 1                   
        
    xyzl_new = np.empty( ( xyzl.shape[0], xyzl.shape[1] + 3 ) )

    lines = np.unique( xyzl_origin[:,line_c_origin] ) 

    x_ref, y_ref, z_ref = xyz_ref[0], xyz_ref[1], xyz_ref[2]

    xyzl_w = np.copy( xyzl )
    xyzl_w[ :, z_c] = 1
    if xyz_w is not None :
        print( 'Weights interpolation ...')
        xyzl_w[:, z_c]  = utl.xyz2xy( ( xyz_w[0], xyz_w[1], xyz_w[2] ), 
              ( xyzl_w[:,x_c], xyzl_w[:,y_c] ), method='cubic', fillnan=True )[0]                                         
        print( 'Done!') 

    print( 'Reference field interpolation ...' )
    xyzl_ref = np.copy( xyzl )
    xyzl_ref[:, z_c] = utl.xyz2xy( ( x_ref, y_ref, z_ref ), 
          ( xyzl_ref[:,x_c], xyzl_ref[:,y_c] ), method='cubic' )[0]  
    print( 'Done!')      
    
    print( 'RemRes loop ...')   

    
    xyzl_out = np.copy( xyzl_origin )
    xyzl_out = np.column_stack( ( xyzl_origin, np.zeros( ( xyzl_origin.shape[0], 2 ) ) ) )  

    for l in lines :
                
        idx = xyzl[ : , line_c ] == l
        idxo = xyzl_origin[ : , line_c_origin ] == l
        line = xyzl[ idx ]
        liner = xyzl_ref[ idx ]
        
        line_w = xyzl_w[ idx ] 
        line_new = np.zeros( line.shape[0] ) 
        line_rem = np.zeros( line.shape[0] ) 
        line_res = np.zeros( line.shape[0] )
        line_ref = np.zeros( line.shape[0] )
           
        lim_lin = utl.xy2lim( line[:,x_c], line[:,y_c], extend=True, d=wind_size )        
        xl_ref, yl_ref, idxr = utl.xy_in_lim( x_ref, y_ref, lim_lin )
        zl_ref = z_ref[ idxr ]
        
        if line.shape[0] > 1 : 
            
            xl, yl, idxl = utl.xy_in_lim( line[:,x_c], line[:,y_c], lim_lin )
            zl = line[ idxl, z_c ] 
                                        
            for i in range( line.shape[0] ) : 
                         
                win_i = utl.neighbors_points( ( xl, yl ), 
                        ( line[i,x_c], line[i,y_c] ), wind_size )[2]     
    
                pl_i = zl[ win_i ]
                
                line_rem[i] = np.nanmean( pl_i ) 

                # win_ii = utl.neighbors_points( ( xl_ref, yl_ref ), 
                #           ( line[i,x_c], line[i,y_c] ), half_ws )[2] 
                         
                # pl_ii = zl_ref[ win_ii ]
                
                win_ii = utl.neighbors_points( ( liner[ :, x_c], liner[ :, y_c] ), 
                        ( line[i,x_c], line[i,y_c] ), half_ws )[2]  
                 
                pl_ii = liner[ win_ii, z_c ]
                
                if np.sum( win_ii ) == 0 :
                    line_ref[i] = 0
                else :    
                    line_ref[i] = np.nanmean( pl_ii )
                       
            line_res = line_ref * line_w[:,z_c] 
            
            line_rem = line_rem * line_w[:,z_c]  
            z_new = line[:,z_c] - line_rem + line_res 
        
        else :
            z_new = line[:,z_c]  
            
        xyzl_new[ idx, 0] = line[:,x_c]
        xyzl_new[ idx, 1] = line[:,y_c]
        xyzl_new[ idx, 2] = z_new
        xyzl_new[ idx, 3] = line[:,line_c]
        xyzl_new[ idx, 4] = liner[:,z_c]
        xyzl_new[ idx, 5] = line[:,z_c] - line_new
        xyzl_new[ idx, 6] = line[:,z_c]
       
        xyzl_out[idxo, -1] = utl.xyz2xy( ( xyzl_new[idx,0], xyzl_new[idx,1], xyzl_new[idx,2] ), 
              ( xyzl_origin[idxo,x_c_origin], xyzl_origin[idxo,y_c_origin] ), method='cubic' )[0] 
         
        xyzl_out[idxo, -2] = utl.xyz2xy( ( xyzl[idx,x_c], xyzl[idx,y_c], xyzl_new[idx,4] ), 
              ( xyzl_origin[idxo,x_c_origin], xyzl_origin[idxo,y_c_origin] ), method='cubic' )[0] 
     
    print( 'Done!')    
    
    z_c_new = xyzl_out.shape[1]-1
    z_c_ref = xyzl_out.shape[1]-2

    if pad_dist != 0 :

        isn = xyzl_out[:,line_c_origin] == pad_idx
        xyzl_out = xyzl_out[ ~isn ]
        
    if filt is not None :
            
        xyzl_out = line_filt( xyzl_out, wind_size=wfilt, prjcode_in=prjcode_in, 
                              filter_type=filt, poly_order=6, prjcode_out=prjcode_in, 
                              x_c=x_c_origin, y_c=y_c_origin, z_c=-1, 
                              line_c=line_c, extend_factor=2, edge_fix=True ) 

    if adjst_lev == True : 
        print( 'Adjust levelling ...') 
        xyzl_out,_,cover = line_levellig( xyzl_out, x_c=x_c_origin, y_c=y_c_origin, z_c=z_c_new, 
                                          line_c=line_c_origin, power=power, iterations=iterations, 
                                          dist=dist, spl_k=spl_k, spl_s=spl_s, order_c=order_c ) 
        print( 'Done!') 
        
    if median_lev == True :
        print( 'Median levelling ...') 
        if median_lines == 'crossing' :
            cover = utl.cross_over_points( xyzl_out, x_c=x_c_origin, y_c=y_c_origin, z_c=z_c_new, 
                          line_c=line_c_origin, method='linear' )[:,3]    
            median_lines = np.unique( np.in1d( xyzl_out[:,line_c_origin], 
                                               np.unique( cover ), invert=True ) ).tolist()  
        if median_lines == [] :  
            median_lines = np.unique( xyzl_out[:,line_c_origin] ).tolist()   
            print( line_c_origin, median_lines )                                     
        xyzl_out, cover = median_levellig( xyzl_out, dist=dist, x_c=x_c_origin, y_c=y_c_origin,
                                    z_c=z_c_new, line_c=line_c_origin, radius=radius, lines=median_lines, 
                                    order_c=order_c ) 
        print( 'Done!') 
        
    if ( ( adjst_lev == False ) and ( lines.size > 1 ) ) or ( ( median_lev == False ) and ( lines.size > 1 ) ):
        
        cover = utl.cross_over_points( xyzl_out, x_c=x_c_origin, y_c=y_c_origin, z_c=z_c_new, 
                      line_c=line_c_origin, method='linear' )
        
        if cover.shape[ 0 ] > 0 :
            minz, maxz, meanz, stdz = utl.stat( cover[ :, 6 ], decimals=2 )
        
            print( 'cross-over error:', 'min =',minz,',', 'max =',maxz,',', 
                                        'mean =',meanz,',', 'std =',stdz )  
    
    # -------------------------------------------------------------------------
    # Plots      
        
    if plot is True :
        
        plt.figure()
        
        if vmin == None :
            vmin = np.nanmean( xyzl_out[:,z_c_new] ) - 2 * np.std( xyzl_out[:,z_c_new] )
        if vmax == None :
            vmax = np.nanmean( xyzl_out[:,z_c_new] ) + 2 * np.std( xyzl_out[:,z_c_new] )          
        
        plt.subplot(1,2,1)
        plt.title('Original')
        plt.scatter( xyzl_origin[:,x_c_origin], xyzl_origin[:,y_c_origin], 
                     s=s, c=xyzl_origin[:,z_c_origin], cmap='rainbow', vmin=vmin, vmax=vmax )
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)         
        
        plt.subplot(1,2,2)
        plt.title('Leveled')
        plt.scatter( xyzl_out[:,x_c_origin], xyzl_out[:,y_c_origin], s=s, c=xyzl_out[:,z_c_new], 
                     cmap='rainbow', vmin=vmin, vmax=vmax )
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)         

        plt.colorbar( ax=plt.gcf().axes, location='bottom', shrink=0.6 )
        
    if plot_cross is True :
        
        plt.figure(figsize=(8, 6))
        
        plt.subplot(1,2,1)
        cover_old = utl.cross_over_points( xyzl_out, plot=False, vmin=vminc, vmax=vmaxc, 
            colorbar=False, method='linear', x_c=x_c_origin, y_c=y_c_origin, 
            z_c=z_c_origin, line_c=line_c_origin )  
        
        Min, Max, Mean, Std = utl.stat( cover_old[:,6], decimals=2 )
        plt.scatter( xyzl_out[:,x_c_origin], xyzl_out[:,y_c_origin], s=s, 
                     marker='_', c='k', alpha=0.5, linewidths=0.05 )
        plt.scatter( cover_old[:,0], cover_old[:,1], s=s*10, c=cover_old[:,6], cmap='rainbow',
                     vmin=vminc, vmax=vmaxc )
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        
        plt.title( 'Cross-Over Error Original : \n' + f'Min={Min}  Max={Max}  Mean={Mean}  Std={Std}' )        

        plt.subplot(1,2,2)
        Min, Max, Mean, Std = utl.stat( cover[:,6], decimals=2 )
        plt.scatter( xyzl_out[:,x_c_origin], xyzl_out[:,y_c_origin], s=s, 
                     marker='_', c='k', alpha=0.5, linewidths=0.05 )
        plt.scatter( cover[:,0], cover[:,1], s=s*10, c=cover[:,6], cmap='rainbow',
                      vmin=vminc, vmax=vmaxc )
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        
        plt.title( 'Cross-Over Error Final : \n' + f' Min={Min}  Max={Max}  Mean={Mean}  Std={Std}' )    

        plt.tight_layout()    

        cbar = plt.colorbar( ax=plt.gcf().axes, location='bottom', shrink=0.6 )     
        plt.text(1.02, 0.5, '[ mGal ]', va='center', ha='left', rotation=0, transform=cbar.ax.transAxes)

        
    if plot_lines == True :
        
        if units == 'degree' : deg2m = True
        else : deg2m = False

        fl = lz_plot.plot_lines( xyzl_out, line_c=line_c_origin, x_c=x_c_origin, y_c=y_c_origin, 
                                 z_c=[z_c_origin,z_c_new,z_c_ref], deg2m=deg2m, plot_points=False, 
                                 marker='+', marker_color='k', s=1.5, x_units=x_units, 
                                 y_units=y_units, c=['b','g','k'], 
                                 legend=[ 'original_line', 'leveled_line', 'reference_line' ] ) 
      
    else : fl = None    
    
    if new_xy is False :
        xyzl_out[:,x_c_origin], xyzl_out[:,y_c_origin] = utl.prjxy( prjcode_out, prjcode_in, 
                                                      xyzl_out[:,x_c_origin], 
                                                      xyzl_out[:,y_c_origin] )     
        
    return  xyzl_out, xyzl_new, xyzl_ref, cover

# -----------------------------------------------------------------------------       
def line_levellig( xyzl, prjcode_in=4326, prjcode_out=4326, dist=None, x_c=0, y_c=1,
                   z_c=2, line_c=3, s=1, iterations=1, plot=False, vmin=None, vmax=None,
                   plot_cross=False, vminc=None, vmaxc=None, deg_to_m=False, power=2,
                   new_xy=True, x_units='', y_units='[mGal]', plot_lines=False,
                   spl_k=3, spl_s=None, order_c=None, lines=[] ) :
                   
    xyzl = np.copy( xyzl )    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = utl.prjxy( prjcode_in, prjcode_out, 
                                              xyzl[:,x_c], xyzl[:,y_c] )
    
    xyzl, idx_original = utl.sort_lines( xyzl, line_c=line_c, x_c=x_c, y_c=y_c, 
                                         add_dist=False, order_c=order_c )
    if lines == [] :
        lines = np.unique( xyzl[ :, line_c ] )
        
    if dist == None :
        dist = utl.lines_samp_dist( xyzl, line_c=line_c, x_c=x_c, y_c=y_c, 
                                    deg_to_m=deg_to_m, kind='mode' ) 
    
    xyzl_new = np.copy( xyzl )
    # -------------------------------------------------------------------------
    # Start itarations
    
    for itr in range( iterations ) :
        
        xyzl_re = utl.resamp_lines( xyzl_new, dist, method='linear', order_c=order_c,
                                    line_c=line_c, x_c=x_c, y_c=y_c, z_c=z_c, lines=lines )
        
        cross_p = utl.cross_over_points( xyzl_re, method='linear', 
                                         x_c=0, y_c=1, z_c=2, line_c=3 )
        
        
        i1 = np.isin( cross_p[:,2], lines ) 
        i2 = np.isin( cross_p[:,3], lines )
        lines_cross = np.unique( np.concatenate( ( cross_p[i1,3], cross_p[i2,2], lines ) ) )
        cross_p = cross_p[ i1 | i2 ]
        
        N = np.size( lines_cross ) # Number of survey lines
        K = []
        W = []
        g = []
        
        for i, l in enumerate( lines_cross ) :
            
            if xyzl[ xyzl[ :, line_c ] == l ].shape[0] <= 1 : continue
            
            idx_c = ( cross_p[:,2] == l ) | ( cross_p[:,3] == l ) 
            cross_pi = cross_p[ idx_c ]
            Delta_g = np.zeros( cross_pi.shape[0] )
            
            if np.size( Delta_g ) == 0 :
                continue
            else :
                K.append( np.size( Delta_g ) ) 
                
                for li in range( K[-1] ) :
                    if cross_pi[li,2] == l :
                        g.append( cross_pi[li] ) 
                    if cross_pi[li,3] == l :  
                        col_change = [0,1,3,2,5,4,6]
                        g.append( cross_pi[ li, col_change ] ) 
                    
                    Delta_g[li] = g[-1][4] - g[-1][5]    
                
                if ( np.sum( Delta_g**2 ) == 0 ) or ( l not in lines ) :
                    w_line = [ K[-1] / 1e-50, l ] 
                else :    
                    w_line = [ K[-1] / np.sum( Delta_g**2 ), l ]                
                W.append( w_line ) 
                
        g = np.array( g )
        W = np.array( W )        
                
        W_star = W[:,0] * ( N / np.sum( W[:,0] ) ) 
        
        g_ij = np.zeros( g.shape[0] )
        C_ij = np.zeros( g.shape[0] )
        
        for n, _ in enumerate( g ) :
            
            W_star_i = W_star[ W[:,1] == g[n,2] ] ** power
            W_star_j = W_star[ W[:,1] == g[n,3] ] ** power
                
#            m = ( W_star_j / W_star_i ) * power
#            g_ij[ n ] =  ( g[n,4] + g[n,5] * m ) / ( 1 + m )   
            g_ij[ n ] = ( g[n,4] * W_star_i + g[n,5] * W_star_j ) / \
                        ( W_star_i + W_star_j )         
            C_ij[ n ] = g_ij[ n ] - g[n,4]
                    
        for l in W[:,1] :
            
            if l in lines :
            
                idx = xyzl_re[:, 3] == l 
                idx_cross = g[:,2] == l
                g_cross = C_ij[ idx_cross ]
                
                line_dists = np.linspace( 0, dist*np.size( xyzl_re[idx, 2] ), 
                                          np.size( xyzl_re[idx, 2] ) )            
                
                if ( np.size( g_cross ) == 1 ) : 
                    g_cross_line = np.repeat( g_cross[0], np.sum( idx ) )
                    
                if ( np.size( g_cross ) >= 2 ) :                 
                    
                    x_start, y_start = xyzl_re[idx, 0][0], xyzl_re[idx, 1][0]  
                    
                    dist_cross =  np.sqrt( ( g[idx_cross,0] - x_start )**2 + 
                                           ( g[idx_cross,1] - y_start )**2 )
                    
                    si = np.argsort( dist_cross )
                    dist_cross = dist_cross[si]
                    g_cross = g_cross[si]
                    int_c_f = utl.sp.interpolate.interp1d( dist_cross, g_cross, kind='nearest',
                              bounds_error=False, fill_value='extrapolate' ) 
                                                      
                    g_cross_line = int_c_f( line_dists )
                    
#                line_dists_desemp = line_dists[::2]
                    # -------------------------------------------------------------
                    # spline 
                    if g_cross_line.size <= spl_k :
                        spl_k2 = g_cross_line.size - 1
                    else : spl_k2 = spl_k   
                    spl = utl.sp.interpolate.UnivariateSpline( line_dists, 
                               g_cross_line, k=spl_k2, s=spl_s ) 
                    g_cross_line = spl( line_dists )  
                    # g_cross_line = utl.sp.ndimage.uniform_filter( g_cross_line, 5 )
                                        
                xyzl_re[idx, 2] = xyzl_re[idx, 2] + g_cross_line   
            
        for i, l in enumerate( lines ) :   
            
            idx_new = xyzl_new[ :, line_c ] == l
            line_new = xyzl_new[ xyzl_new[ :, line_c ] == l ]
            
            if line_new.shape[0] <= 1 : continue
            
            line_re = xyzl_re[ xyzl_re[ :, 3 ] == l ]
            
            dist_re = utl.geo_line_dist( line_re[:,0], line_re[:,1], order='same' )[0]  
            dist_new = utl.geo_line_dist( line_new[:,x_c], line_new[:, y_c], order='same' )[0] 
            
            if dist_re.size > 1 :
                int_new = utl.sp.interpolate.interp1d( dist_re, line_re[:, 2], 
                          kind='linear', bounds_error=False, fill_value='extrapolate' )   
                xyzl_new[idx_new, z_c] = int_new( dist_new )
            else :
              xyzl_new[idx_new, z_c] = line_re[:, 2] 
            
        
        cover = utl.cross_over_points( xyzl_new, vmin=vminc, vmax=vmaxc, method='linear',
                                       x_c=x_c, y_c=y_c, z_c=z_c, line_c=line_c )  
        
        minz, maxz, meanz, stdz = utl.stat( cover[ :, 6], decimals=2 )
    
        print( 'iteration ', itr+1, ':\n', 
               'cross-over error:', 'min =',minz,',', 'max =',maxz,',', 
                                    'mean =',meanz,',', 'std =',stdz )        
    
    xyzl_new = xyzl_new[ idx_original, : ]
    xyzl = xyzl[ idx_original, : ]
    
    # -------------------------------------------------------------------------
    #Plots
    if plot is True :
        
        plt.figure()
        
        if vmin == None :
            vmin = np.nanmean( xyzl_new[:,z_c] ) - 2 * np.std( xyzl_new[:,z_c] )
        if vmax == None :
            vmax = np.nanmean( xyzl_new[:,z_c] ) + 2 * np.std( xyzl_new[:,z_c] )          
        
        plt.subplot(1,2,1)
        plt.title('Original')
        plt.scatter( xyzl[:,x_c], xyzl[:,y_c], s=s, c=xyzl[:,z_c], cmap='rainbow',
                     vmin=vmin, vmax=vmax )
        
        plt.subplot(1,2,2)
        plt.title('Leveled')
        plt.scatter( xyzl_new[:,x_c], xyzl_new[:,y_c], s=s, c=xyzl_new[:,z_c], cmap='rainbow',
                     vmin=vmin, vmax=vmax )
        
        plt.tight_layout()
        plt.colorbar( ax=plt.gcf().axes, location='bottom', shrink=0.6 )    
    
    if plot_cross is True :
        
        plt.figure()

        plt.subplot(1,2,2)
        cover = utl.cross_over_points( xyzl_new, plot=True, vmin=vminc, vmax=vmaxc, method='linear',
                                       colorbar=False, x_c=x_c, y_c=y_c, z_c=z_c, line_c=line_c )  

        if vminc == None :
            vminc = np.min( cover[:,6] ) 
        if vmaxc == None :
            vmaxc = np.max( cover[:,6] )     
        
        plt.subplot(1,2,1)
        _ = utl.cross_over_points( xyzl, plot=True, vmin=vminc, vmax=vmaxc, method='linear',
                                   colorbar=False, x_c=x_c, y_c=y_c, z_c=z_c, line_c=line_c )       

        plt.tight_layout()
        plt.colorbar( ax=plt.gcf().axes, location='bottom', shrink=0.6 )
        
    if plot_lines == True :
        
        if deg_to_m == True : deg2m = True
        else : deg2m = False
        
        z_old_c = xyzl.shape[1]
        xyzl_new = np.column_stack( ( xyzl_new, xyzl[ :, z_c ] ) )
        fl = lz_plot.plot_lines( xyzl_new, line_c=line_c, x_c=x_c, y_c=y_c, z_c=[z_old_c,z_c], 
                              deg2m=deg2m, plot_points=False, marker='+', marker_color='k',
                              s=1.5, x_units=x_units, y_units=y_units, c=['b','g' ], 
                              legend=[ 'original_line', 'leveled_line' ] )  
        
        xyzl_new = np.delete( xyzl_new, z_old_c, 1 )
        
    else : fl = None    
    
    if new_xy is False :
        xyzl_new[:,x_c], xyzl_new[:,y_c] = utl.prjxy( prjcode_out, prjcode_in, 
                                                      xyzl_new[:,x_c], 
                                                      xyzl_new[:,y_c] )         
        
    return xyzl_new, fl, cover
        
        
# -----------------------------------------------------------------------------       
def median_levellig( xyzl, prjcode_in=4326, prjcode_out=4326, dist=None, x_c=0, y_c=1,
                     z_c=2, line_c=3, s=1, iter=1, plot=False, vmin=None, vmax=None,
                     new_xy=True, plot_lines=False,radius=None, lines=[], 
                     deg_to_m=False, order_c=None ) :  
                     
    
    xyzl = np.copy( xyzl )    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = utl.prjxy( prjcode_in, prjcode_out, 
                                              xyzl[:,x_c], xyzl[:,y_c] )
    
    xyzl, idx_original = utl.sort_lines( xyzl, line_c=line_c, x_c=x_c, y_c=y_c, 
                                         add_dist=False, order_c=order_c )
    
    if dist == None :
        dist = utl.lines_samp_dist( xyzl, line_c=line_c, x_c=x_c, y_c=y_c, 
                                    deg_to_m=deg_to_m, kind='mode' ) 
    
    xyzl_new = np.copy( xyzl )
    
    # -------------------------------------------------------------------------
    # Start itarations
    if radius is None :
        radius = [ dist*12, dist*6, dist*3 ]
    if type( radius ) in ( int, float ) :
        radius = [ radius for k in range( iter ) ]
    
    for itr in range( len(radius) ) :
        
        xyzl_re = utl.resamp_lines( xyzl, dist, method='linear', order_c=order_c,
                                    line_c=line_c, x_c=x_c, y_c=y_c, z_c=z_c ) 
        
        if lines is [] :
            lines = np.unique( xyzl_re[ :, 3 ] )
        
        for i, l in enumerate( lines ) :
            
            line_new = xyzl_new[ xyzl_new[ :, line_c ] == l ]
            
            if line_new.shape[0] < 1 : continue
            
            lim_lin = utl.xy2lim( line_new[:,x_c], line_new[:,y_c], extend=True, 
                                  d=radius[itr]*2 )        
            xl_re, yl_re, idx_re = utl.xy_in_lim( xyzl_re[:,0], xyzl_re[:,1], lim_lin )
            zl_re = xyzl_re[ idx_re, 2 ] 
            line_re = xyzl_re[ idx_re, 3 ]            

            for ip, p in enumerate( line_new ) :
                
#                win_i = ( xyzl_re[:,0] > line_new[ip,x_c] - radius[itr] ) & \
#                        ( xyzl_re[:,0] < line_new[ip,x_c] + radius[itr] ) & \
#                        ( xyzl_re[:,1] > line_new[ip,y_c] - radius[itr] ) & \
#                        ( xyzl_re[:,1] < line_new[ip,y_c] + radius[itr] ) 
                        
                win_i = utl.neighbors_points( ( xl_re, yl_re ), 
                        ( line_new[ip,x_c], line_new[ip,y_c] ), radius[itr] )[2]                                                                  
                        
                Am = np.nanmedian( zl_re[ win_i ] )
                Lm = np.median( zl_re[ win_i ][ line_re[win_i] == l] )
                line_new[ip, z_c] = line_new[ip, z_c] + Am - Lm
        
            xyzl_new[ xyzl_new[:,line_c] == l ] = line_new
                
    #Plots

        cover = utl.cross_over_points( xyzl_new, x_c=x_c, y_c=y_c, z_c=z_c, 
                    line_c=line_c, method='linear' )

        print( cover.shape )
        minz, maxz, meanz, stdz = utl.stat( cover[ :, 6], decimals=2 )

        print( 'iteration ', itr+1, ':\n', 
                'cross-over error:', 'min =',minz,',', 'max =',maxz,',', 
                                    'mean =',meanz,',', 'std =',stdz )    
    
    if plot is True :
        plot_xyzl_o = xyzl[ np.in1d( xyzl[:,line_c], lines ) ]
        plot_xyzl_n = xyzl_new[ np.in1d( xyzl_new[:,line_c], lines ) ]        
        plt.figure()
        
        if vmin == None :
            vmin = np.nanmean( plot_xyzl_o[:,z_c] ) - 2 * np.std( plot_xyzl_o[:,z_c] )
        if vmax == None :
            vmax = np.nanmean( plot_xyzl_o[:,z_c] ) + 2 * np.std( plot_xyzl_o[:,z_c] )          
        
        plt.subplot(1,3,1)
        plt.title('Original')
        plt.scatter( plot_xyzl_o[:,x_c], plot_xyzl_o[:,y_c], s=s, c=plot_xyzl_o[:,z_c], cmap='rainbow',
                     vmin=vmin, vmax=vmax )
        
        plt.subplot(1,3,2)
        plt.title('Leveled')
        plt.scatter( plot_xyzl_n[:,x_c], plot_xyzl_n[:,y_c], s=s, c=plot_xyzl_n[:,z_c], cmap='rainbow',
                     vmin=vmin, vmax=vmax )
        plt.colorbar()
        
        plt.subplot(1,3,3)
        plt.title('Difference')
        plt.scatter( plot_xyzl_n[:,x_c], plot_xyzl_n[:,y_c], s=s, c=plot_xyzl_o[:,z_c]-plot_xyzl_n[:,z_c], 
                    cmap='rainbow' )  
        plt.colorbar()
                           
        
        plt.tight_layout()
#        plt.colorbar( ax=plt.gcf().axes, location='bottom', shrink=0.6 )    
    
    if new_xy is False :
        xyzl_new[:,x_c], xyzl_new[:,y_c] = utl.prjxy( prjcode_out, prjcode_in, 
                                                      xyzl_new[:,x_c], 
                                                      xyzl_new[:,y_c] )         
        
    return xyzl_new, cover  


# -----------------------------------------------------------------------------
def drift( stations, time, gobs, deg=1, dtime='datetime64', plot=False ) :

    # Creating empty variables to append the results
    drift = []
    time_drift = []
    labels= []

    # Creating numpy arrays with the input variables
    stations = np.array( stations )
    time = np.array( time )
    gobs = np.array( gobs )

    # Sorting the inmput variables by time
    idx = time.argsort()
    stations = stations[ idx ]
    time = time[ idx ]
    gobs = gobs[ idx ]

    # Convert time array to float number with timestamp (i.e., seconds from 01/01/1970)
    if dtime == 'datetime64' :
        time_num = (time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    if dtime == 'seconds' :
        time_num = time
         
    # Starting loop to compute the drift fuction
    for i, r in enumerate( gobs ) :

            idx = stations[ :i ] == stations[ i ]
            if np.sum( idx ) != 0 :
                if len(drift) == 1 :
                    drift.append( gobs[i] - gobs[:i][idx][0] )   

                else :
                    print( i )
                    f = utl.sp.interpolate.interp1d( time_drift, drift, kind='linear')
                    dlv = f( time_num[:i][idx][0] )

                    if time_num[:i][idx][0] not in time_drift :
                        drift.append( dlv )
                        time_drift.append( time_num[:i][idx][0]  )
                        labels.append( stations[:i][idx][0] )   
                    drift.append( gobs[i] - ( gobs[:i][idx][0] - dlv ) )                     
                
                time_drift.append( time_num[i]  )
                labels.append( stations[i] ) 

            else :
                if i == 0 :
                    drift.append( 0 )               
                    time_drift.append( time_num[i]  )
                    labels.append( stations[i] )

    # Creating numpy arrays with the output variables
    labels = np.array( labels )
    time_drift = np.array( time_drift )
    drift = np.array( drift )

    # Sorting the output variables by time
    idx = time_drift.argsort()
    labels = labels[ idx ]
    time_drift = time_drift[ idx ]
    drift = drift[ idx ]

    dlp = np.polyfit( time_drift, drift, deg=deg )
    drift_curv = np.polyval( dlp, time_num )    



    time_drift = np.array( [ dt.datetime.utcfromtimestamp( i ) for i in time_drift ] )

    if plot is True : 

        plt.scatter( time_drift, drift )
        plt.plot( time_drift, drift, linestyle='dashed' )
        plt.plot( time_drift, drift_curv )
        for i, txt in enumerate(labels):
            plt.annotate(txt, (time_drift[i], drift[i]))

    return drift_curv, time_drift
            
# -----------------------------------------------------------------------------
def earth_tides( lat, lon, z=0, gtime=None, 
                 yy=None, mm=None, dd=None, h=None, m=None, s=None ) :

    if gtime is None :

        gtime = utl.combine64( years=yy, months=mm, days=dd, 
                          hours=h, minutes=m, seconds=s ) 
        
    if type( gtime ) == np.ndarray :
        gtime = gtime.tolist()

    if ( type( lat ) in ( int, float ) ) and ( type( lon ) in ( int, float ) ) and \
        ( np.size(gtime) > 1 ) :

        lat = np.full( np.size(gtime), lat )
        lon = np.full( np.size(gtime), lon )

    if ( type( lat ) not in ( int, float ) ) and ( type( z ) in ( int, float ) ) :
        z = np.full( np.size(lat), z )        

    if ( type( lat ) in ( int, float ) ) and ( type( lon ) in ( int, float ) ) and \
        ( np.size(gtime) == 1 ) :
    
        model = TideModel() 
        tides = model.solve_longman(lat, lon, z, gtime )[2]

    else :

        tides = np.full( np.size(lat), np.nan )
        for i,_ in enumerate( tides ) :
                    model = TideModel() 
                    tides[i] = model.solve_longman(lat[i], lon[i], z[i], gtime[i] )[2]
    
    return tides
    
# -----------------------------------------------------------------------------
class TideModel():
    """Class to encapsulate the Longman 1959 tide model."""

    def __init__(self):
        """Initialize the model object."""
        self.name = 'Model'
        self.results = namedtuple('results', ['model_time', 'gravity_moon',
                                              'gravity_sun', 'gravity_total'])
        self.results.model_time = []
        self.results.gravity_moon = []
        self.results.gravity_sun = []
        self.results.gravity_total = []

    def calculate_julian_century(self, timestamp):
        """Calculate the julian century and hour.
        Take a datetime object and calculate the decimal Julian century and
        floating point hour. This is in reference to noon on December 31,
        1899 as stated in the Longman paper.
        Parameters
        ----------
        timestamp: datetime
            Time stamp to convert
        Returns
        -------
        float, float
            Julian century and hour
        """
        origin_date = datetime(1899, 12, 31, 12, 00, 00)  # Noon Dec 31, 1899
        dt = timestamp - origin_date
        days = dt.days + dt.seconds / 3600. / 24.
        decimal_julian_century = days / 36525
        julian_hour = (timestamp.hour + timestamp.minute / 60. +
                       timestamp.second / 3600.)
        return decimal_julian_century, julian_hour

    def solve_longman(self, lat, lon, alt, time):
        """Solve the tide model.
        Given the location and datetime object, computes the current
        gravitational tide and associated quantities. Latitude and longitude
        and in the traditional decimal notation, altitude is in meters, time
        is a datetime object.
        Parameters
        ----------
        lat : float
            latitude (in degrees)
        lon : float
            longitude (in degrees)
        alt : float
            altitude (in meters)
        time : datetime
            time at which to solve the model
        Returns
        -------
        float, float, float
            lunar, solar, and total gravitational tides
        """
        T, t0 = self.calculate_julian_century(time)

        if t0 < 0:
            t0 += 24.
        if t0 >= 24:
            t0 -= 24.

        mu = 6.673e-8  # Newton's gravitational constant
        M = 7.3537e25  # Mass of the moon in grams
        S = 1.993e33  # Mass of the sun in grams
        e = 0.05490  # Eccentricity of the moon's orbit
        m = 0.074804  # Ratio of mean motion of the sun to that of the moon
        # Mean distance between the centers of the earth and the moon
        c = 3.84402e10
        # Mean distance between centers of the earth and sun in cm
        c1 = 1.495e13
        h2 = 0.612  # Love parameter
        k2 = 0.303  # Love parameter
        a = 6.378270e8  # Earth's equitorial radius in cm
        i = 0.08979719  # (i) Inclination of the moon's orbit to the ecliptic
        # Inclination of the Earth's equator to the ecliptic 23.452 degrees
        omega = np.radians(23.452)
        # For some reason his lat/lon is defined with W as + and E as -
        L = -1 * lon
        lamb = np.radians(lat)  # (lambda) Latitude of point P
        H = alt * 100.  # (H) Altitude above sea-level of point P in cm

        # Lunar Calculations
        # (s) Mean longitude of moon in its orbit reckoned
        # from the referred equinox
        s = (4.72000889397 + 8399.70927456 * T + 3.45575191895e-05 * T * T +
             3.49065850399e-08 * T * T * T)
        # (p) Mean longitude of lunar perigee
        p = (5.83515162814 + 71.0180412089 * T + 0.000180108282532 * T * T +
             1.74532925199e-07 * T * T * T)
        # (h) Mean longitude of the sun
        h = 4.88162798259 + 628.331950894 * T + 5.23598775598e-06 * T * T
        # (N) Longitude of the moon's ascending node in its orbit
        # reckoned from the referred equinox
        N = (4.52360161181 - 33.757146295 * T + 3.6264063347e-05 * T * T +
             3.39369576777e-08 * T * T * T)
        # (I) Inclination of the moon's orbit to the equator
        I = np.arccos(np.cos(omega)*np.cos(i) - np.sin(omega)*np.sin(i)*np.cos(N))
        # (nu) Longitude in the celestial equator of its intersection
        # A with the moon's orbit
        nu = np.arcsin(np.sin(i)*np.sin(N)/np.sin(I))
        # (t) Hour angle of mean sun measured west-ward from
        # the place of observations
        t = np.radians(15. * (t0 - 12) - L)

        # (chi) right ascension of meridian of place of observations
        # reckoned from A
        chi = t + h - nu
        # cos(alpha) where alpha is defined in eq. 15 and 16
        cos_alpha = np.cos(N) * np.cos(nu) + np.sin(N) * np.sin(nu) * np.cos(omega)
        # sin(alpha) where alpha is defined in eq. 15 and 16
        sin_alpha = np.sin(omega) * np.sin(N) / np.sin(I)
        # (alpha) alpha is defined in eq. 15 and 16
        alpha = 2 * np.arctan(sin_alpha / (1 + cos_alpha))
        # (xi) Longitude in the moon's orbit of its ascending
        # intersection with the celestial equator
        xi = N - alpha

        # (sigma) Mean longitude of moon in radians in its orbit
        # reckoned from A
        sigma = s - xi
        # (l) Longitude of moon in its orbit reckoned from its ascending
        # intersection with the equator
        l = (sigma + 2 * e * np.sin(s - p) + (5. / 4) * e * e * np.sin(2 * (s - p)) +
             (15. / 4) * m * e * np.sin(s - 2 * h + p) + (11. / 8) *
             m * m * np.sin(2 * (s - h)))

        # Sun
        # (p1) Mean longitude of solar perigee
        p1 = (4.90822941839 + 0.0300025492114 * T + 7.85398163397e-06 *
              T * T + 5.3329504922e-08 * T * T * T)
        # (e1) Eccentricity of the Earth's orbit
        e1 = 0.01675104 - 0.00004180 * T - 0.000000126 * T * T
        # (chi1) right ascension of meridian of place of observations
        # reckoned from the vernal equinox
        chi1 = t + h
        # (l1) Longitude of sun in the ecliptic reckoned from the
        # vernal equinox
        l1 = h + 2 * e1 * np.sin(h - p1)
        # cosine(theta) Theta represents the zenith angle of the moon
        cos_theta = (np.sin(lamb) * np.sin(I) * np.sin(l) + np.cos(lamb) * (np.cos(0.5 * I)**2
                     * np.cos(l - chi) + np.sin(0.5 * I)**2 * np.cos(l + chi)))
        # cosine(phi) Phi represents the zenith angle of the run
        cos_phi = (np.sin(lamb) * np.sin(omega) * np.sin(l1) + np.cos(lamb) *
                   (np.cos(0.5 * omega)**2 * np.cos(l1 - chi1) +
                   np.sin(0.5 * omega)**2 * np.cos(l1 + chi1)))

        # Distance
        # (C) Distance parameter, equation 34
        C = np.sqrt(1. / (1 + 0.006738 * np.sin(lamb)**2))
        # (r) Distance from point P to the center of the Earth
        r = C * a + H
        # (a') Distance parameter, equation 31
        aprime = 1. / (c * (1 - e * e))
        # (a1') Distance parameter, equation 31
        aprime1 = 1. / (c1 * (1 - e1 * e1))
        # (d) Distance between centers of the Earth and the moon
        d = (1. / ((1. / c) + aprime * e * np.cos(s - p) + aprime * e * e *
             np.cos(2 * (s - p)) + (15. / 8) * aprime * m * e * np.cos(s - 2 * h + p)
             + aprime * m * m * np.cos(2 * (s - h))))
        # (D) Distance between centers of the Earth and the sun
        D = 1. / ((1. / c1) + aprime1 * e1 * np.cos(h - p1))

        # (gm) Vertical componet of tidal acceleration due to the moon
        gm = ((mu * M * r / (d * d * d)) * (3 * cos_theta**2 - 1) + (3. / 2) *
              (mu * M * r * r / (d * d * d * d)) *
              (5 * cos_theta**3 - 3 * cos_theta))
        # (gs) Vertical componet of tidal acceleration due to the sun
        gs = mu * S * r / (D * D * D) * (3 * cos_phi**2 - 1)

        love = (1 + h2 - 1.5 * k2)
        g0 = (gm + gs) * 1e3 * love
        return gm * 1e3 * love, gs * 1e3 * love, g0

    def run_model(self):
        """Run the model for a range of times.
        Runs the tidal model beginning at start_time with time steps of
        increment seconds for days.
        """
        self.n_steps = int(24 * self.duration * 3600 / self.increment)

        for i in np.arange(self.n_steps):
            time_at_step = (self.start_time +
                            i * timedelta(seconds=self.increment))
            gm, gs, g = self.solve_longman(self.latitude, self.longitude,
                                           self.altitude, time_at_step)
            self.results.model_time.append(time_at_step)
            self.results.gravity_moon.append(gm)
            self.results.gravity_sun.append(gs)
            self.results.gravity_total.append(g)

    def plot(self):
        """Plot the model results.
        Make a simple plot of the gravitational tide results from the
        model run.
        """
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(111)
        ax1.set_xlabel(r'Date', fontsize=18)
        ax1.set_ylabel(r'Anomaly [mGal]', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.plot_date(self.results.model_time, self.results.gravity_total,
                      '-k', linewidth=2)
        plt.show()
        return fig, ax1

    def write(self, fname):
        """Write model results to file.
        Write results out of a file for later analysis or reading into another
        method for analysis/correction of data.
        Parameters
        ----------
        fname: string
            name of file to save
        """
        t_string = datetime.strftime(self.start_time, '%Y-%m-%dT%H:%M:%S')
        f = open(fname, 'w')
        f.write('Station latitude: {self.latitude}\n')
        f.write('Station longitude: {self.longitude}\n')
        f.write('Station altitude [m]: {self.altitude}\n')
        f.write('Time Increment [s]: {self.increment}\n')
        f.write('Start Time: {t_string}\n')
        f.write('Duration [days]: {self.duration}\n')
        f.write('\nTime,Lunar,Solar,Total\n')
        f.write('YYYY-MM-DDTHH:MM:SS\tmGal\tmGal\tmGal\n')

        for i in np.arange(self.n_steps):
            t_string = datetime.strftime(self.results.model_time[i],
                                         '%Y-%m-%dT%H:%M:%S')
            f.write('{}\t{}\t{}\t{}\n'.format(t_string,
                                              self.results.gravity_moon[i],
                                              self.results.gravity_sun[i],
                                              self.results.gravity_total[i]))
        f.close()

# -----------------------------------------------------------------------------
def convert2burrisfmt( station, date, g, lon, lat, 
                       elev=0, tide_corr=0, meter='D45', 
                       oper='abc', feedback=0, dial_setting=50000,
                       path_name='burris.dat', level_corr=0, 
                       temp_corr=0, beam_err=0, height=0 ) :
    
   if np.issubdtype( date.dtype, np.datetime64 ) : 
        date = np.datetime_as_string( date, unit='s')
        date = np.char.replace( date, '-', '/' )
        date = np.char.replace( date, 'T', ' ' )

   if type( lon ) in ( int, float ) :
        lon = np.full( np.size(g), lon )

   if type( lat ) in ( int, float ) :
        lat = np.full( np.size(g), lat ) 

   if type( elev ) in ( int, float ) :
        elev = np.full( np.size(g), elev ) 

   if type( oper ) in ( int, float, str ) :
        oper = np.full( np.size(g), oper ) 

   if type( meter ) in ( int, float, str ) :
        meter = np.full( np.size(g), meter )  

   if type( feedback ) in ( int, float, str ) :
        feedback = np.full( np.size(g), feedback ) 

   if type( dial_setting ) in ( int, float, str ) :
        dial_setting = np.full( np.size(g), dial_setting ) 

   if type( tide_corr ) in ( int, float, str ) :
        tide_corr = np.full( np.size(g), tide_corr ) 

   if type( level_corr ) in ( int, float, str ) :
        level_corr = np.full( np.size(g), level_corr )  

   if type( temp_corr ) in ( int, float, str ) :
        temp_corr = np.full( np.size(g), temp_corr )  

   if type( beam_err ) in ( int, float, str ) :
        beam_err = np.full( np.size(g), beam_err )  

   if type( height ) in ( int, float, str ) :
        height = np.full( np.size(g), height )         

   with open( path_name, 'w' ) as f : 
        for i, gi in enumerate( g ) :
             f.write( f"{station[i]} {oper[i]} {meter[i]} "+ 
                      f"{date[i]} {gi:>.5f} {dial_setting[i]} {feedback[i]} "+ 
                      f"{tide_corr[i]} {level_corr[i]} {temp_corr[i]} {beam_err[i]} {height[i]} "+
                      f"{elev[i]} {lat[i]:.5f} {lon[i]:.5f} \n" )
             
   f.close()

   return path_name