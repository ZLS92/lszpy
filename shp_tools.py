# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 17:05:39 2020

@author: zampa
"""

import os 
mdir = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(1, mdir)

import glob
from osgeo import gdal, ogr, osr
import numpy as np
from matplotlib import pyplot as plt, path as mpath, patches as mpatches
import utils as utl

OGRTypes = { int: ogr.OFTInteger, str: ogr.OFTString, float: ogr.OFTReal,
             'str_': ogr.OFTString, 'float64':ogr.OFTReal,
             'int64':ogr.OFTInteger } 

# -----------------------------------------------------------------------------
def write_points( x, y, fields=None, name='points_shp', path=None, prj_code=4326, 
                  kml=True, csv=True, write_xy=True ): 
    """
    Function to crete a .shp file of points given the points coordinates
    
    x, y --> coordinates of points (it works if x and y are numpy_arrays, but it should works also with lists or tuples)
    fields --> dictionary with attributes of each point 
               e.g fields={'height':heigh_of_points, 'height_error':h_err, ...} 
               (with 'heigh_of_points' and 'h_err' being numpy_arrays... but it should works also with lists or tuples)
    name --> name of the new file (NB. the file is alwas created within a folder with its name) 
    path --> path where the new/file_folder is created   
    prj_code --> code from Proj library to identify the coordinate the reference systhem (default is geo. WGS84, i.e. 4326)
                 (if its an int. it will be recognized as an epsg code)   
    kml, csv --> if True a copy of the file issaved also as GoogleEarth or Com.Sep.Val. formats 
    
    NB. THIS FUNTION IS BASED ON GDAL/OGR Python API (first tested whith gdal version 3.0.0)     
         
    """

    # Define Driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    
    if path is None: 
        shp_ = name
    else: 
        shp_ = path + os.sep + name 

    pt_nmS = shp_ + os.sep + name + '.shp'   
    
    os.makedirs(shp_, exist_ok=True)  

    while True:
        try:             
            if os.path.exists( pt_nmS ) :
                os.remove( pt_nmS )
        
            # Define Driver
            driver = ogr.GetDriverByName("ESRI Shapefile")
            # Create shp file
    
            point_data_source = driver.CreateDataSource(pt_nmS)
            # Define srs
            srs = osr.SpatialReference()
            srs.ImportFromWkt( utl.prj_(prj_code).to_wkt() )
            # Define layer name and type
            point = point_data_source.CreateLayer(name, srs, ogr.wkbPoint)
            # Add fields
            if write_xy == True :
                xField = ogr.FieldDefn('x', ogr.OFTReal)
                point.CreateField(xField)
                yField = ogr.FieldDefn('y', ogr.OFTReal)
                point.CreateField(yField) 
            if fields is not None:
                for f in fields: 
                    Field = ogr.FieldDefn(f, OGRTypes[type(fields[f][0]).__name__]) 
                    point.CreateField(Field)         
            # Create the feature and set values
            featureDefn = point.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            pointi = ogr.Geometry(ogr.wkbPoint)

            for n, (i, j) in enumerate(zip(x,y)):
                pointi.AddPoint(float(i), float(j))
                feature.SetGeometry(pointi)
                if write_xy == True :
                    feature.SetField("x", i)
                    feature.SetField("y", j)
                if fields is not None:
                    for f in fields:
                       feature.SetField(f, fields[f][n])
                point.CreateFeature(feature)             

            del point_data_source 

            if kml==True: # Save as kml in the same folder
                points_kml = gdal.VectorTranslate(shp_ + os.sep + name + '.kml' ,pt_nmS, format='KML')
                del points_kml

            if csv == True :

                fields['shp_x'] = x
                fields['shp_y'] = y
                utl.dict2csv( fields, path_name=shp_ + os.sep + name + '.csv'  )

            return pt_nmS  
        
        except Exception as e: 
            print(e)
            del point_data_source
            
# -----------------------------------------------------------------------------
def write_ply(xy_rings, fields=None, name='points_shp', path=None, prj_code=4326, kml=True, csv=False): 
    
    """
    Function to crete a .shp file of polygons given the polygons boudaries coordinates
    
    x, y --> coordinates of points (it works if x and y are numpy arrays, it should works also with lists or tuples)
    fields --> dictionary with attributes of each point 
               e.g fields={'height':heigh_of_points, 'height_error':h_err, ...} 
               (with 'heigh_of_points' and 'h_err' being numpy arrays, it should works also with lists or tuples)
    name --> name of the new file (NB. the file is alwas created within a folder with its name) 
    path --> path where the new/file_folder is created   
    prj_code --> code from proj library to identify the coordinate reference systhem 
                 (if its an int. it will be recognized as an epsg code)   
    kml, csv --> if True they allawed to save a copy of the file also in GoogleEarth or Com.Sep.Val. formats 
    
    NB. THIS FUNTION IS BASED ON GDAL/OGR Python API (first tested whith gdal version 3.0.0)     
         
    """    

    if path is None: 
        shp_ = name
    else: 
        shp_ = path + os.sep + name 

    ply_nmS = shp_ + os.sep + name + '.shp'
    os.makedirs( shp_, exist_ok=True )    
          
    # Define Driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # Create shp file
    while True:
        try:      
            os.remove( ply_nmS ) # check if file is used in another process
            ply_data_source = driver.CreateDataSource(ply_nmS)
            # Define srs 
            srs = osr.SpatialReference()
            srs.ImportFromWkt(utl.prj_(prj_code).to_wkt())
            # Define layer name and type
            ply = ply_data_source.CreateLayer(name, srs, ogr.wkbMultiPolygon)
            # Add fields
            id_field = ogr.FieldDefn('id', ogr.OFTInteger)
            ply.CreateField(id_field) 
            if fields is not None:
                for f in fields: 
                    Field = ogr.FieldDefn(f, OGRTypes[type(fields[f][0]).__name__]) 
                    ply.CreateField(Field)        
                    
            # Create the feature and set values
            featureDefn = ply.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            feature_geom = ogr.Geometry(ogr.wkbPolygon)
            ring_geom = ogr.Geometry(ogr.wkbLinearRing)
            
            for n, ring in enumerate(xy_rings):
                for r in range(0,ring.shape[0]):
                    ring_geom.AddPoint(ring[r,0], ring[r,1])
                feature_geom.AddGeometry(ring_geom)    
                feature.SetGeometry(feature_geom) 
                feature.SetField('id', n) 
                if fields is not None:
                    for f in fields:
                          feature.SetField(f, fields[f][n])  
                ply.CreateFeature(feature)       
            
            del ply_data_source
            if kml==True: # Save as kml in the same folder
                gdal.VectorTranslate(shp_ + os.sep + name + '.kml' ,ply_nmS, format='KML')
                
            return ply_nmS  
        
            break
        except Exception as e: 
            print(e)
            del ply_data_source  
            
# -----------------------------------------------------------------------------
def xy_in_ply( x, y, input_ply, proj_xy=4326, within=True, plot=False ) :
    
    proj_ply = shp_prjcode( input_ply, p=False )
    xp, yp = utl.prjxy( proj_xy, proj_ply, x, y )
    
    lim = utl.xy2lim( xp, yp, extend=True, d=10 )
    input_ply = translate( input_ply, lim=lim )
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds_in = driver.Open(input_ply)  
    lyr_in = ds_in.GetLayer(0) 
    
    mask = np.full(np.size(x), False, dtype=bool)
    pt = ogr.Geometry(ogr.wkbPoint)
    for feat_in in lyr_in:
      ply = feat_in.GetGeometryRef()
      if type(ply) != None.__class__:
          minx,maxx,miny,maxy = ply.GetEnvelope()
          menv = np.where( (xp>minx) & (xp<maxx) & (yp>miny) & (yp<maxy) )[0]
          for n in menv:
                mask[n] = pt.SetPoint_2D(0, xp.ravel()[n], yp.ravel()[n])
                mask[n] = pt.Within(ply)
            
    if within is True : 
        xf, yf, xdiff, ydiff = x[mask], y[mask], x[~mask], y[~mask]   
    if within is False : 
        xf, yf, xdiff, ydiff = x[~mask], y[~mask], x[mask], y[mask] 
        mask = ~mask
    
    del ds_in
    
    if plot==True:
        plt.scatter( xdiff, ydiff, c='b', label='Non_Selected' )
        plt.scatter( xf, yf, c='r', label='Selected' )
        plt.legend()       
    
    return xf, yf, mask

# -----------------------------------------------------------------------------
def shp_lim( input_shp, layer=0, extend=False, d=10, method='percentage', 
             sqr_area=False, plot=False ) :
    
    ds_in = ogr.Open( input_shp )  
    lyr_in = ds_in.GetLayer( layer ) 

    lim = lyr_in.GetExtent()
    
    if extend == True :
        lim = utl.extend_lim( lim, d=d, method=method, sqr_area=sqr_area, plot=plot )
             
    if ( plot is True ) and ( extend is False ) :
        xplot = [lim[0], lim[0], lim[1], lim[1], lim[0]]
        yplot = [lim[2], lim[3], lim[3], lim[2], lim[2]]
        plt.plot(xplot, yplot, c='k')
        
    ds_in = None
    lyr_in = None
    
    return lim    
# -----------------------------------------------------------------------------
def ply_bounds( input_ply, ply_id=0, prj_code=4326 ):
    
    ds_in = ogr.Open(input_ply)  
    lyr_in = ds_in.GetLayer(0) 
    
    for n, feat_in in enumerate(lyr_in):
      ply = feat_in.GetGeometryRef()
      minx,maxx,miny,maxy = ply.GetEnvelope()
      if n==ply_id: break
    
    proj_ply = lyr_in.GetSpatialRef().ExportToProj4()
    xp, yp = utl.prjxy(proj_ply, prj_code, [minx,maxx], [miny,maxy])
    bounds = [xp[0], xp[1], yp[0], yp[1]]
    
    ds_in = None
    lyr_in = None
    
    return bounds
    
# -----------------------------------------------------------------------------
def lim2ring(lim, prj_in=4326, prj_out=4326):
    
    if prj_in!=prj_out:
        [lim[0], lim[1]], [lim[2],lim[3]] = utl.prjxy(prj_in, prj_out, [lim[0], lim[1]], [lim[2],lim[3]])

    ring = np.zeros((5,2))  
    ring[0,:] = np.array((lim[0], lim[3]))
    ring[1,:] = np.array((lim[0], lim[2]))   
    ring[2,:] = np.array((lim[1], lim[2]))   
    ring[3,:] = np.array((lim[1], lim[3]))    
    ring[4,:] = np.array((lim[0], lim[3]))  

    return ring    

# -----------------------------------------------------------------------------
def plot_ply( shp_ply, facecolor='r', edgecolor='k', lim=None,
              plot=True, new_fig=False, ax=None, 
              prjcode_out=4326, prjcode_in=None, alpha=0 ):
    
    # Extract first layer of features from shapefile using OGR
    
    if prjcode_in is None :
        prjcode_in = shp_prjcode( shp_ply, p=False ) 

    if lim is not None :
        shp_ply = translate( shp_ply, new_name=None, new_path='/vsimem/', 
                             suffix='_cut' , extension='shp', lim=lim )

    ds = ogr.Open( shp_ply )
    lyr = ds.GetLayer(0)
    
    plyco = []
    lyr.ResetReading()
    
    for feat in lyr:
        geom = feat.GetGeometryRef()
        codes = []
        all_x = []
        all_y = []
        
        for i in range(geom.GetGeometryCount()):
            # Read ring geometry and create path
            ref = geom.GetGeometryRef(i)
            x = [ref.GetX(j) for j in range(ref.GetPointCount())]
            y = [ref.GetY(j) for j in range(ref.GetPointCount())]
            # skip boundary between individual rings
            codes += [mpath.Path.MOVETO] + (len(x)-1)*[mpath.Path.LINETO]                    
            all_x += x
            all_y += y
        # path = mpath.Path(np.column_stack((all_x,all_y)), codes)
        # paths.append( path )
        plyco.append( [ x, y ] )

        if prjcode_in != prjcode_out :
            x, y = utl.prjxy( prjcode_in, prjcode_out, x, y )
            plyco[-1] = [ x, y ]
        
        # Add paths as patches to axes
        if plot is True :
            if new_fig is True :
                plt.figure()
            for ply in plyco:
                if ax is None :
                    ax = plt.gca() 
                if facecolor is not None :
                    ax.fill(ply[0], ply[1], color=facecolor, alpha=alpha )
                ax.plot( ply[0], ply[1], color=edgecolor ) 
                plt.show()
        
        ds = None
        
        return ply

# -----------------------------------------------------------------------------
def ogr2ogr(in_shp, arguments, new_name=None, new_path='/vsimem/', suffix='', extension='shp'):
    
    if new_path == None:
        new_path = os.path.dirname(os.path.realpath(in_shp))
    
    if new_path != '/vsimem/':
        os.makedirs(new_path, exist_ok=True)
        new_path = new_path + os.sep
    
    if new_name == None:
        new_name = in_shp.split(os.sep)[-1].split('.')[0] + '_new'
        
    out_shp = new_path + new_name + suffix +'.'+extension
    
    cmd = f'ogr2ogr {arguments} {out_shp} {in_shp}'
    
    os.system(cmd)
    
    return out_shp
    
# -----------------------------------------------------------------------------     
def shp_prjcode( shp, p=False ) :
    
    if type( shp ) == str :
        shp = ogr.Open( shp )

    lyr = shp.GetLayer()
    code = lyr.GetSpatialRef()
    if code is None:
        print('Could not determine SRID')
        prjcode = None
    else :
        prjcode = utl.prj_( code.ExportToProj4() ).srs 
    
    if p == True :
        print( prjcode )
    
    shp = None
    lyr = None    
    
    return prjcode 

# -----------------------------------------------------------------------------   
def shp_epsg( shp, p=False ) :
    
    if type( shp ) == str :
        shp = ogr.Open( shp )

    lyr = shp.GetLayer()
    code = lyr.GetSpatialRef()

    # Try to determine the EPSG/SRID code
    if code.AutoIdentifyEPSG() == 0: # success
       epsg = int( code.GetAuthorityCode(None) ) 
    else:
        print('Could not determine SRID')
    
    if p == True :
        print( epsg )
    
    shp = None
    lyr = None    
    
    return epsg

# -----------------------------------------------------------------------------
def shp_name( shp, p=False ) :
    
    if type( shp ) == str :
        shp = ogr.Open( shp ) 
        
    path_name = shp.GetDescription()
    path_name = path_name.replace("//", "/")
    path_name = path_name.replace("/", "\\")
    name = path_name.split('\\')[-1].split('.')[0]
    
    if p == True :
        print( name )    
    
    shp = None
    
    return name

# -----------------------------------------------------------------------------
def shp_path( shp, p=False ) :
    
    if type( shp ) == str :
        shp = ogr.Open( shp ) 
        
    path_name = shp.GetDescription()
    path_name = path_name.replace( "//", "/" )
    path_name = path_name.replace( "/", "\\" )
    name = path_name.split( "\\" )[-1]
    pn_split = list(shp.GetDescription())
    n_split = list(name)
    n_len = len(n_split) 
    del pn_split[-n_len::] 
    
    while ( pn_split[-1] == '/' ) or ( pn_split[-1] == '\\' ) :
        del pn_split[-1]
    
    path = ''.join( pn_split )    

    shp = None
    
    if p == True :
        print( path )     
    
    return path

# -----------------------------------------------------------------------------
def translate( in_shp, 
               new_name=None, 
               new_path='/vsimem/', 
               suffix='_new' ,
               extension='shp',
               lim=None,
               lim_prjcode=None,
               clip2lim = True,
               in_prjcode=None, 
               out_prjcode=None, 
               options = '',
               buffer=None ) :
    
    """    
    in_shp --- a .shp Dataset object or a filename
    new_name --- name of the output file (if None, it will be the same of the in_shp + suffix = '_new')
    new_path --- path of the output file (default is the file system '/vsimem/'; if None, it will be the same of the in_shp )
    suffix --- suffix to add after the name (e.g., Coast.shp --> suffix = '_new' --> output_name = Coast_new.shp)     
    lim --- spatial filter as (minX, minY, maxX, maxY) bounding box
    lim_prjcode --- SRS in which the lim is expressed. If not specified, it is assumed to be the one of the layer(s)
    options --- can be be an array of strings, a string or let empty and filled from other keywords.
    fmt --- output format ("ESRI Shapefile", etc...)
    accessMode --- None for creation, 'update', 'append', 'overwrite'
    in_prjcode --- source SRS
    out_prjcode --- output SRS (with reprojection if reproject = True)
    skipFailures --- whether to skip failures 
    
    See also GDAL/OGR Python API ( https://gdal.org/python/ )
    """

    if type ( in_shp ) is not str :
        in_shp = in_shp.GetDescription()
        
    if new_name == None :
        new_name = shp_name( in_shp ) + suffix   
        
    if new_path == None :
        new_path = shp_path( in_shp )
    else :
        if new_path != '/vsimem/' :
           new_path = new_path + os.sep + new_name
           os.makedirs( new_path, exist_ok=True )
           new_path = new_path + os.sep

    out_shp = new_path + new_name + '.' + extension    
    
    if out_prjcode is None : 
        out_prjcode = shp_prjcode( in_shp ) 
    else :
        out_prjcode = utl.prj_( out_prjcode ).srs   
    
    options = options + f' -t_srs "{out_prjcode}"'    
   
    if in_prjcode is None : 
        in_prjcode = shp_prjcode( in_shp )     
    else :
        in_prjcode = utl.prj_( in_prjcode ).srs    

    options = options + f' -s_srs "{in_prjcode}"'  
        
    if lim is not None :
        if lim_prjcode is None :
            lim_prjcode = in_prjcode
        if utl.prj_(lim_prjcode).srs != utl.prj_(out_prjcode).srs :    
            lim = utl.prj_lim( lim, lim_prjcode, out_prjcode )
        lim = utl.lim_sort( lim )    

    if ( lim is not None ) and ( clip2lim == True ) :
        options = options + f' -clipdst {lim[0]} {lim[1]} {lim[2]} {lim[3]} '
        
    if buffer is not None : 
        in_shp_name = shp_name( in_shp )
        options = options + f' -dialect sqlite -sql "select ST_Buffer(geometry, {buffer}) from {in_shp_name}"'
    
#    print( options )
    new_shp = gdal.VectorTranslate( out_shp, in_shp, options = options ) 
    
    if new_path != '/vsimem/' :
        new_shp = None

    return out_shp


# -----------------------------------------------------------------------------
#def createBuffer( in_shp, bufferDist, new_name=None, new_path='/vsimem/', suffix='_new' ):
#    
#    if type ( in_shp ) != str :
#        in_shp = in_shp.GetDescription()
#        
#    if new_name == None :
#        new_name = shp_name( in_shp ) + suffix   
#        
#    if new_path == None :
#        new_path = shp_path( in_shp )
#    else :
#        if new_path != '/vsimem/' :
#           new_path = new_path + os.sep + new_name
#           os.makedirs( new_path, exist_ok=True )
#           new_path = new_path + os.sep
#
#    out_shp = new_path + new_name + '.shp'      
#    
#    if type ( in_shp ) != str :
#        inputds = ogr.Open(in_shp)
#        
#    inputlyr = inputds.GetLayer()
#
#    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
#    
#    if os.path.exists( out_shp ):
#        shpdriver.DeleteDataSource( out_shp )
#        
#    outputBufferds = shpdriver.CreateDataSource(out_shp)
#    bufferlyr = outputBufferds.CreateLayer(out_shp, geom_type=ogr.wkbPolygon)
#    featureDefn = bufferlyr.GetLayerDefn()
#
#    for feature in inputlyr:
#        ingeom = feature.GetGeometryRef()
#        geomBuffer = ingeom.Buffer(bufferDist)
#
#        outFeature = ogr.Feature( featureDefn )
#        outFeature.SetGeometry( geomBuffer )
#        bufferlyr.