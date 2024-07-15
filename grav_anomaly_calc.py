# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:55:41 2019

@author: lzampa
"""
import numpy as np
import os
import lszpy.utils as utl

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
    
    a = 978032.67714
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
def slab(h, dc=2670, dw=1030, topo_sea=False):
    """
    Gravity effect of a flat slab [mGal]
    h = thickness    
    d = density    
    """
    
    if topo_sea == True:
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