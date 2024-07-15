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
gmp = imp.load_source( 'module.name', mdir+os.sep+'gravmag_processing.py' )
fa = imp.load_source( 'module.name', mdir+os.sep+'gobs2fa.py' )
te_hm = imp.load_source( 'module.name', mdir+os.sep+'te_harmonica.py' )

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
abs_path = os.path.abspath(__file__)
mpath = os.path.dirname(abs_path)

# -----------------------------------------------------------------------------
def gobs2bg( gdb, gs1, gs2, R1, R2=None, dtm1_list=None, dtm2_list=None, 
             InOut_ply1=[], InOut_ply2=[], new_dtm=False, coast=None, herr=2.5,
             name='New_Gravity_DataBase', new_path="", x_c=1, y_c=0, g_c=2, h_c=3, 
             st_type_c=None, ref_c=None, line_c=None, suffix='_Corr&An' ) :

    print(__name__)
    # -------------------------------------------------------------------------
    # Load database
    if type(gdb) == str:
        path = os.path.dirname(gdb)
        name = os.path.basename(gdb).split('.')[0]
        gdb = np.loadtxt( gdb, delimiter=',', skiprows=1 ) 
    else: 
        path = mpath    
    
    if new_path != "": 
        path = new_path+s+name
        os.makedirs(path, exist_ok=True) 
    
    # -------------------------------------------------------------------------
    # Create Te_correction directory    
    Te_correction_dir = path +s+ 'Te_correction'
    os.makedirs( Te_correction_dir, exist_ok=True )   
    
    # -------------------------------------------------------------------------
    # Extract columns 
    lon = gdb[:,x_c]
    lat = gdb[:,y_c]
    gobs = gdb[:,g_c]
    
    # -------------------------------------------------------------------------
    # Create high res. and low res. Terrein Bathymetric Models (dtm1 and dtm2)
    # Merge raster files

    print('===================================================================\n'
          'Create / Load Digital Topographic Bathymetric Models (i.e. Te_dtm1 & Te_dtm2)\n'
          'It may take a while ...')
    
    if type(dtm1_list) == str:
        dtm1_dir = dtm1_list
        dtm1_list = os.listdir(dtm1_list)    
    else: 
        dtm1_dir = path +s+ 'Te_correction' +s+ 'dtm1'
        dtm1_list = os.listdir(dtm1_dir) 
    
    if type(dtm2_list) == str:
        dtm2_dir = dtm2_list
        dtm2_list = os.listdir(dtm2_list)
    else: 
        dtm2_dir = path +s+ 'Te_correction' +s+ 'dtm2' 
        dtm2_list = os.listdir(dtm2_dir)

    for file in dtm1_list+dtm2_list:
        if file.endswith(".shp"):
            coast = file
    
    dtm1_str = dtm1_dir +s+ 'dtm1.tif'
    if dtm1_list is None: 
        dtm1_list= utl.absolute_file_paths(dtm1_dir)
    
    if ((os.path.isfile(dtm1_str) is False) or (new_dtm is True)) and (len(dtm1_list)>1):
        
        lim_dtm1 = utl.xy_lim(lon, lat, extend=True, d=utl.m2deg(R1))       
        dtm1 = rt.merge( dtm1_list, 
                         lim = lim_dtm1, 
                         min_res = utl.m2deg( gs1 ), 
                         final_res = None,
                         nodata = 9999,
                         prjcode = 4326, 
                         lim_prjcode = 4326, 
                         int_attribute = InOut_ply1,
                         name = 'dtm1',
                         path = path+s+'Te_correction',  
                         shp_ply = coast,                           
                         plot = False )        
         
    if ( len(dtm1_list)==1 ) and ( os.path.isfile(dtm1_str) is False ):
        dtm1 = dtm1_list[0]
    else: 
        dtm1 = dtm1_str    
    os.makedirs(Te_correction_dir+s+'dtm1', exist_ok=True)    
    rt.gdal_save(dtm1, new_name='dtm1', new_path=Te_correction_dir+s+'dtm1', extension='tif', close=True)[1]         

    dtm2_str = dtm2_dir +s+ 'dtm2.tif'
    if dtm2_list is None: 
        dtm2_list= utl.absolute_file_paths(dtm2_dir)
    
    if ((os.path.isfile(dtm2_str) is False) or (new_dtm is True)) and (len(dtm2_list)>1):
        
        lim_dtm2 = utl.xy_lim(lon, lat, extend=True, d=utl.m2deg(168000))
        dtm1 = rt.merge( dtm2_list, 
                         lim = lim_dtm2, 
                         min_res = utl.m2deg( gs2 ), 
                         final_res = None,
                         nodata = 9999,
                         prjcode = 4326, 
                         lim_prjcode = 4326, 
                         int_attribute = InOut_ply2,
                         name = 'dtm2',
                         path = path+s+'Te_correction',  
                         shp_ply = coast,                           
                         plot = False)          
        
    if (len(dtm2_list)==1) and (os.path.isfile(dtm2_str) is False):
        dtm2 = dtm2_list[0]
    else: 
        dtm2 = dtm2_str    
    os.makedirs(path+s+'Te_correction'+s+'dtm2', exist_ok=True)    
    rt.gdal_save(dtm2, new_name='dtm2', new_path=Te_correction_dir+s+'dtm2', extension='tif', close=True)[1]
    
    print('Done! \n'
          '===================================================================\n'
          'Compute gravity corrections and anomalies \n'
          '...') 
    
    # -------------------------------------------------------------------------
    # Free Air Anomly 
    gdb_fa = fa.gobs2fa( gdb, dtm=dtm1, herr=7, name=name, path=path,
                         x_c=x_c, y_c=y_c, g_c=g_c, h_c=h_c, st_type_c=st_type_c, 
                         ref_c=ref_c, line_c=line_c, suffix=suffix )
    dict = gdb_fa[0]
    path_name = gdb_fa[1]
    array = gdb_fa[2]
    headers = gdb_fa[3]
    fmt = gdb_fa[4]
    
    hnew = dict['H_New']
    st_type = dict['St_Type']
    gn = dict['Gn']
    a_c = dict['Atm_C']
    faw_c = dict['Faw_C']
    
    # -------------------------------------------------------------------------
    # Topo Correction
    D_Te = te_hm.run_te(lon, lat, st_type=st_type, dtm1=dtm1, dtm2=dtm2, 
                                             z=hnew, gs1=gs1, R1=R1, gs2=gs2, R2=R2, 
                                             herr=herr, output_file=path+s+'Te_correction',
                                             cpu=None)[1]    
    Te, te_near, te_far, ist = D_Te['te_tot'], D_Te['te_near'], D_Te['te_far'], \
                               D_Te['st_num'] 
    # -------------------------------------------------------------------------
    #  Bouguer anomaly
    bg_a = gobs - ( gn + a_c + faw_c + Te )         

    print('Done! \n'   
          '=================================================================== \n'
          'Save Data \n'
          '...')    
    # ------------------------------------------------------------------------
    # Save 
    array = np.column_stack( ( array, Te, bg_a ) )
    headers.append('Te')
    headers.append('Bg_A')
    fmt.append('% 12.3f')
    fmt.append('% 12.3f')
    utl.array2csv( array, headers=headers, sep=',', fmt=fmt, 
                   path_name=path_name )    

# -----------------------------------------------------------------------------
#if __name__ == '__main__':
#    
#    gdb = input('Set DataSet absolute path_name: \n')
#    gs = input('Set Grid_Reslution1 and Grid_Reslution2 in meters (e.g. 100 500): \n').split()
#    gs1, gs2 = [float(i) for i in gs]
#    R = input('Set Grid_Radius1 and Grid_Radius2 in meters (e.g. 10000 90000): \n').split() 
#    R1, R2 = [float(i) for i in R]
#    new_path = input('Set Output directory (press Enter to set the same of input DS):\n')       
#    dtm1_list = input('Set directory with High Resolution DTMs (perss Enter to the default "../dtm1"):\n')  
#    if dtm1_list == "": 
#        dtm1_list = mpath+s+'dtm1'
#    dtm2_list = input('Set directory with Low Resolution geotif DTMs (perss Enter to the default "../dtm2"):\n')       
#    if dtm2_list == "": 
#        dtm2_list = mpath+s+'dtm2'   
#             
#    gobs2bg(gdb, gs1, gs2, R1, R2=R2, dtm1_list=dtm1_list, dtm2_list=dtm2_list, 
#            herr=2.5, new_path=new_path)
 