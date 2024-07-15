# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:41:12 2021

@author: zampa
"""

import numpy as np
import utils as utl
import matplotlib.pyplot as plt

G = 6.67299 * 1e-11 # [m3/(kg *s^2)]


# -----------------------------------------------------------------------------
def xyz2prisms( x, y, z, bottom=0, step=None, size=None, plot=False, grid=True,
                method="nearest" ) :
    
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    z = np.array(z).ravel()
    
    if type( z ) in ( int, float ) :
        z = np.repeat( z, len(x) )
    
    X, Y, step = utl.xy2XY( x, y, step=step, grid=grid, return_step=True )
    
    xp = X.ravel()
    yp = Y.ravel()
    
    if np.size( X.ravel() ) != np.size( x ) :
        Z,_ = utl.xyz2xy( ( x, y, z ), ( xp, yp ), method=method )
        zp = Z.ravel()
    else :
        zp = z
    
    if type( bottom ) in ( int, float ) :
        bottom = np.repeat( bottom, np.size(xp) )

    eb = np.array( xp ) - step[0]/2
    wb = np.array( xp ) + step[0]/2
    sb = np.array( yp ) - step[1]/2
    nb = np.array( yp ) + step[1]/2
    lb = np.array( bottom )
    ub = np.array( zp )

    prisms = np.column_stack( ( eb, wb, sb, nb, lb, ub ) )
    
    if plot is True :
        
        plt.figure() 
        plt.scatter( eb, sb, s=size, c='r', marker='+')
        plt.scatter( eb, nb, s=size, c='r', marker='+')
        plt.scatter( wb, sb, s=size, c='r', marker='+')
        plt.scatter( wb, nb, s=size, c='r', marker='+')
        plt.scatter( xp, yp, s=size, c=zp )
    
    return prisms


# =============================================================================
class nagy() :
    
    def __init__( self, coordinates, prism, density ) :
        
        self.coordinates = coordinates
        self.prism = prism
        self.density = density
        
        if type( prism ) in ( list, tuple ) :
            self.prism = np.asarray( prism )   
     
        if type( coordinates ) in ( list, tuple ) :
            self.coordinates = np.array( coordinates, ndmin=2 )      
            
        if np.size( density ) == 1 :
            self.density = np.repeat( density, self.prism.shape[0] )
            
        if type( density ) in ( list, tuple ) :
            self.density = np.asarray( density )              
            
    # ------------------------------------------------------------------------- 
    def gz( self ) :
        
        np.seterr(divide='ignore', invalid='ignore')
        
        self.pgz = np.zeros( self.coordinates.shape[0] )
        
        for i, c in enumerate( self.coordinates ) :
            
            C = np.tile( self.coordinates[i,:], ( self.prism.shape[0], 1 ) )
            
            array = np.column_stack( ( C, self.prism, self.density ) ) 
            
            # self.pgz[i] = np.sum( np.apply_along_axis( nagy_gz, 1, array ) )
            self.pgz[i] = np.sum( nagy_gz(array) )
            
        return self.pgz 
    
# -----------------------------------------------------------------------------   
def nagy_fz( x, y, z ):

    r = ( x**2 + y**2 + z**2 )**( 1/2 )  
    
    div = np.nan_to_num( ( x * y ) / ( z * r )  ) 
    logyr = np.nan_to_num( np.log( y + r ) )  
    logxr = np.nan_to_num( np.log( x + r ) ) 
    # ---

    fzi = x * logyr + y * logxr - ( z * np.arctan( div ) )

    return fzi

# -----------------------------------------------------------------------------  
def nagy_gz( array ):
    
    x0, y0, z0 = array[:,0], array[:,1], array[:,2]
    
    x1, x2, y1, y2, z1, z2 = array[:,3]-x0, array[:,4]-x0, \
                             array[:,5]-y0, array[:,6]-y0, \
                             array[:,7]-z0, array[:,8]-z0 
    
    rho = array[:,9]                         
    
    gzi = ( ( ( nagy_fz( x2, y2, z2 ) - nagy_fz( x1, y2, z2 ) )
             -( nagy_fz( x2, y1, z2 ) - nagy_fz( x1, y1, z2 ) ) )
           -( ( nagy_fz( x2, y2, z1 ) - nagy_fz( x1, y2, z1 ) )
             -( nagy_fz( x2, y1, z1 ) - nagy_fz( x1, y1, z1 ) ) ) )
    
    gzi = G * rho * gzi *1e5
    
    
    return gzi

# ----------------------------------------------------------------------------- 
                
            







