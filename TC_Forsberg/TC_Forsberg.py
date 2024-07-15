# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:20:07 2019

@author: lzampa
"""
import numpy as np
import os
from osgeo import gdal

#%% Parameters

G = 6.6742*1e-11 # [m3/(kg *s^2)]
M = 5.97*1e24 # [kg]
a_wgs84 = 6378137 # [m]
c_wgs84 = 6356758 # [m]
R_wgs84 = ((a_wgs84**2)*c_wgs84)**(1/3) # [m]
J2_wgs84 = 1.081874*1e-3
w_wgs84 = 7.292115*1e-5 # [rad/sec]

#%% 
def Tc_Forsberg(tc_folder_path, lon, lat, h, array1, array2, latlon1, latlon2, res1, res2, 
                st_number='Nan', R1=5.420, R2=166.7, density=2.670, height='h', calc='tc'):
    
    G1_data = array1
    G1_data[G1_data>1e4]=9999
    G2_data = array2
    G2_data[G2_data>1e4]=9999
    
    gri_name_1  = 'Grid_inner_zone.gri'
    gri_path_name_1 = tc_folder_path +os.sep+ gri_name_1    
    minx_1 = latlon1[0][0]
    maxy_1 = latlon1[0][1]
    maxx_1 = latlon1[1][0]
    miny_1 = latlon1[1][1]   
    header_1 = f'''{round(miny_1,6)} {round(maxy_1,6)} {round(minx_1,6)} {round(maxx_1,6)} {round(res1,6)} {round(res1,6)}'''
    np.savetxt(gri_path_name_1, G1_data, delimiter=' ', fmt='\t%8.2f', header=header_1, comments='')
    
    gri_name_2  = 'Grid_outer_zone.gri'
    gri_path_name_2 = tc_folder_path +os.sep+ gri_name_2   
    minx_2 = latlon1[0][0]
    maxy_2 = latlon1[0][1]
    maxx_2 = latlon1[1][0]
    miny_2 = latlon1[1][1]
    header_2 = f'''{round(miny_2,6)} {round(maxy_2,6)} {round(minx_2,6)} {round(maxx_2,6)} {round(res2,6)} {round(res2,6)}'''
    np.savetxt(gri_path_name_2, G2_data, delimiter='', fmt='\t%7.2f', header=header_2, comments='')    
    
    if st_number=='Nan':
        st_number = np.arange(len(lon)) 
        
    temp_xyz = np.column_stack((st_number, lat, lon, h))     
    np.savetxt(tc_folder_path +os.sep+'measurements.txt', temp_xyz, fmt='% 8d\t %03.3f\t %03.5f\t %04.3f')
    
    ikind = []
    if calc == 'tc':
        ikind = 3
    if calc == 'te':
        ikind = 1

    izcode = []
    if height == 'h':
        izcode = 1
    if height == 'dtm':
        izcode = 0             
        
    with open (tc_folder_path +os.sep+'default_te.ini', 'w') as rsh:
        rsh.write(
f'''measurements.txt             
{gri_name_1}
{gri_name_2}
dummy
Te_correction.txt
1 {ikind} {izcode} 1
{density}
{round(miny_2,5)} {round(maxy_2,5)} {round(minx_2,5)} {round(maxx_2,5)}
{R1} {R2}''')    
    
    H_dir = os.getcwd()           
    os.chdir(tc_folder_path)  
    os.system(f'''tc2018_LZ<default_te.ini''')
    os.chdir(H_dir)
    
    T_data = np.loadtxt(tc_folder_path +os.sep+'Te_correction.txt')
        
    return T_data

#%% 
def Tc_Forsberg_gdal(tc_folder_path, lon, lat, h, dtm1, dtm2, st_number=None,
                     R1=6, R2=166.7, res1=None, res2=None, density=2.670, 
                     height='h', calc='tc', m2deg=True):
    
    """  
    Topographic Effect --------------------------------------------------------
    Simple Bouguer slab + Topographic correction + Curvature correction = Topo effect [te]
    Using gdal raster files as input for topo 
     """   
    R1_deg = R1*360/(2*np.pi*R_wgs84)
    R2_deg = R2*360/(2*np.pi*R_wgs84)
    res1_deg = res1*360/(2*np.pi*R_wgs84)
    res2_deg = res2*360/(2*np.pi*R_wgs84)

    minx, maxx, miny, maxy = np.min(lon),np.max(lon),np.min(lat),np.max(lat)
    lonlat1 = [minx-2*R1_deg, miny-2*R1_deg, maxx+2*R1_deg, maxy+2*R1_deg]      
    dtm1=gdal.Warp('/vsimem/new_gdal_/dtm1.vrt', dtm1, dstSRS='epsg:4326',
                   outputBounds=lonlat1, outputBoundsSRS='epsg:4326',
                    xRes=res1_deg, yRes=res1_deg)
    
    lonlat2 = [minx-R2_deg-2*R1_deg, miny-R2_deg-2*R2_deg, 
               maxx+R2_deg+2*R1_deg, maxy+R2_deg+2*R2_deg]
    dtm2=gdal.Warp('/vsimem/new_gdal_/dtm2.vrt', dtm2, dstSRS='epsg:4326',
                   outputBounds=lonlat2, outputBoundsSRS='epsg:4326',
                   xRes=res2_deg, yRes=res2_deg)
    
    G1_data = dtm1.GetRasterBand(1).ReadAsArray()
    G2_data = dtm2.GetRasterBand(1).ReadAsArray()
    
    gri_name_1  = 'Grid_inner_zone.gri'
    gri_path_name_1 = tc_folder_path +os.sep+ gri_name_1    
    geoTransform = dtm1.GetGeoTransform()
    xRes_1 = geoTransform[1]
    yRes_1 = geoTransform[5]
    xSize_1 = dtm1.RasterXSize
    ySize_1 = dtm1.RasterYSize
    minx_1 = geoTransform[0]
    maxy_1 = geoTransform[3]
    maxx_1 = minx_1+(xRes_1/2) + xRes_1 * (xSize_1-1)
    miny_1 = maxy_1+(yRes_1/2) + yRes_1 * (ySize_1-1)    
    header_1 = f'''{round(miny_1,5)} {round(maxy_1,5)} {round(minx_1,5)} {round(maxx_1,5)} {round(xRes_1,5)} {round(-yRes_1,5)}'''
    np.savetxt(gri_path_name_1, G1_data, delimiter=' ', fmt='\t%8.2f', header=header_1, comments='')
    
    gri_name_2  = 'Grid_outer_zone.gri'
    gri_path_name_2 = tc_folder_path +os.sep+ gri_name_2   
    geoTransform = dtm2.GetGeoTransform()
    xRes_2 = geoTransform[1]
    yRes_2 = geoTransform[5]
    ySize_2 = dtm2.RasterYSize
    xSize_2 = dtm2.RasterXSize
    minx_2 = geoTransform[0]
    maxy_2 = geoTransform[3]
    maxx_2 = minx_2+(xRes_2/2) + xRes_2 * (xSize_2-1)
    miny_2 = maxy_2+(yRes_2/2) + yRes_2 * (ySize_2-1)
    header_2 = f'''{round(miny_2,5)} {round(maxy_2,5)} {round(minx_2,5)} {round(maxx_2,5)} {round(xRes_2,5)} {round(-yRes_2,5)}'''
    np.savetxt(gri_path_name_2, G2_data, delimiter=' ', fmt='\t%8.2f', header=header_2, comments='')     
    
    if st_number is not None:
        st_number = np.arange(len(lon)) 
        
    temp_xyz = np.column_stack((st_number, lat, lon, h))     
    np.savetxt(tc_folder_path +os.sep+'measurements.txt', temp_xyz, fmt='% 8d\t %03.3f\t %03.5f\t %04.3f')
    
    ikind = []
    if calc == 'tc': ikind = 3
    if calc == 'te': ikind = 1

    izcode = []
    if height == 'h': izcode = 1
    if height == 'dtm': izcode = 0             
        
    with open (tc_folder_path +os.sep+'default_te.ini', 'w') as rsh:
        rsh.write(
f'''measurements.txt             
{gri_name_1}
{gri_name_2}
dummy
Te_correction.txt
1 {ikind} {izcode} 1
{density}
{round(miny_2,5)} {round(maxy_2,5)} {round(minx_2,5)} {round(maxx_2,5)}
{R1/1000} {R2/1000}''')    
    
    H_dir = os.getcwd()           
    os.chdir(tc_folder_path)  
    os.system(f'''tc2018_LZ<default_te.ini''')
    os.chdir(H_dir)
    
    T_data = np.loadtxt(tc_folder_path +os.sep+'Te_correction.txt')
    tf = T_data[:,5]
    
    return tf, T_data 
    