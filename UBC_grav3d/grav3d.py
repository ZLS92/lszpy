# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:48:59 2020

@author: lzampa
"""

import os
import numpy as np
import platform
import shutil
import copy
import matplotlib.pyplot as plt
s = os.sep
path_file_py = __file__
mdir =  os.path.dirname( path_file_py )

if platform.system() == 'Windows' :
    pre = ''
if platform.system() == 'Linux' :
    pre = 'wine'

def xyz_sort( xyz ):
    
    yxz = np.copy(xyz)
    yxz[:,[0, 1]] = xyz[:,[1, 0]]
    yxz = yxz[ np.lexsort( ( -yxz[ :, 2 ], yxz[ :, 1 ], yxz[ :, 0 ] ) ) ]   
    
    return yxz

# -----------------------------------------------------------------------------
def xyz2step( x, y, z, round_num=2, step='same' ):
  
    ux = np.sort( np.unique( x ) )
    uy = np.sort( np.unique( y ) )
    uz = np.sort( np.unique( z ))[::-1]
    
    NE = len( ux )
    NN = len( uy )
    NV = len( uz )    
    
    N = (NE, NN, NV)
    
    if step == 'mean' :
        stepx = np.mean( np.abs(np.diff( ux )) ).round(round_num)
        stepy = np.mean( np.abs(np.diff( uy )) ).round(round_num)
        stepz = np.mean( np.abs(np.diff( uz )) ).round(round_num)  
        
        steps = stepx, stepy, stepz
        
        DEn = np.repeat(stepx, NE)
        DNn = np.repeat(stepy, NN)
        DVn = np.repeat(stepz, NV)        
        
    if step == 'same' :
        x, y, z = px2nodes( x, y, z )
        ux = np.sort( np.unique( x ) )
        uy = np.sort( np.unique( y ) )
        uz = np.sort( np.unique( z ))[::-1]
        
        NE = len( ux )
        NN = len( uy )
        NV = len( uz )    
        
        N = (NE, NN, NV)        
        
        DEn = np.abs(np.diff( ux )).round(round_num)
        DNn = np.abs(np.diff( uy )).round(round_num)
        DVn = np.abs(np.diff( uz )).round(round_num)  
        steps = DEn[0], DNn[0], DVn[0] 
        
    Dn = DEn, DNn, DVn
    
    return N, steps, Dn   

# -----------------------------------------------------------------------------
def xyz2llt(x,y,z, round_num=2):
    
    ux = np.unique( x )
    uy = np.unique( y )
    uz = np.unique( z )
    steps = xyz2step( x, y, z, round_num=round_num )[1]
    E0 = np.round( ux.min() - steps[0] / 2 , round_num )
    N0 = np.round( uy.min() - steps[1] / 2 , round_num )
    V0 = np.round( uz.max() + steps[2] / 2 , round_num )    

    return E0, N0, V0

# -----------------------------------------------------------------------------
def create_mesh(Dn, llt, path=mdir, name='mesh'): 
    
    os.makedirs(path, exist_ok=True)
    
    NE = len( Dn[0] ) 
    NN = len( Dn[1] ) 
    NV = len( Dn[2] ) 
    
    E0 = llt[0] 
    N0 = llt[1] 
    V0 = llt[2] 
    
    path_name = path + s + name
    with open(path_name, 'w') as f:
        f.write(f'{NE} {NN} {NV} \n')
        f.write(f'{E0} {N0} {V0} \n')
        f.write( "\n".join( " ".join( map( str, x ) ) for x in ( Dn[0], Dn[1], Dn[2] )))
    
    return  path_name    

# -----------------------------------------------------------------------------
def xyz2mesh(x, y, z, path=mdir, name='mesh', round_num=0):
    
    os.makedirs(path, exist_ok=True)
    
    Dn = xyz2step( x, y, z, round_num=round_num )[2]
    llt = xyz2llt( x, y, z, round_num=round_num )
    mesh = create_mesh( Dn, llt, path=path, name=name )
    
    return mesh

# -----------------------------------------------------------------------------
def create_modelden(d, path=mdir, out_file='model.den', mask_den=None):
    
    if type(mask_den) != None.__class__ :
        d[ mask_den ] = -100.00
    
    model_den = path + s + out_file
    np.savetxt( model_den, d, fmt='% 4.4f' )
    
    return model_den

# -----------------------------------------------------------------------------
def xyz2model_den(x, y, z, d, path=mdir, name_den='model.den', name_mesh='mesh', 
                  round_num=2, nan_val=-100.00, val2nan=None, m_open=False ):
    
    os.makedirs(path, exist_ok=True)
    
    d[ np.isnan(d) ] = nan_val
    if val2nan is not None:
        d[ d==val2nan ] = nan_val
        
    yxzd = xyz_sort( np.column_stack( ( x, y, z, d ) ) )
    y, x, z, d = np.hsplit( yxzd , 4 )
    mesh = xyz2mesh( x, y, z, path=path, name=name_mesh, round_num=round_num )

    model_den = create_modelden(d, path=path, out_file=name_den)

    if path != mdir :
        if os.path.isfile( path +s+ 'MeshTools3d.exe' ) is False:
            shutil.copy2( mdir +s+ 'MeshTools3d.exe', path +s+ 'MeshTools3d.exe' )
    
    if m_open is True:
        meshtools3d( mesh, model_den )
    
    return model_den, mesh, yxzd
    
# -----------------------------------------------------------------------------
def meshtools3d( mesh, model_den ):
    
    meshtools3d = 'MeshTools3d.exe'
    
    path = s.join( mesh.split(s)[:-1] )
    
    if path != mdir :
        if os.path.isfile( path +s+ meshtools3d ) is False:
            shutil.copy2( mdir +s+ meshtools3d, path +s+ meshtools3d )
    
    os.system(f'{pre} {path +s+ meshtools3d} "{mesh}" "{model_den}"')

# -----------------------------------------------------------------------------
def create_bound_den(lb, ub, path=mdir, name_bounds='bounds.den', mask_den=None):
    
    os.makedirs(path, exist_ok=True)
    
    if type(mask_den) != None.__class__ :
        lb[ mask_den ] = -101.00
        ub[ mask_den ] = -100.00
    
    lb[ np.isnan(lb) ] = -101.00
    ub[ np.isnan(ub) ] = -100.00
    
    lub = np.column_stack((lb, ub))
    
    bounds = path + s + name_bounds
    np.savetxt( bounds, lub, fmt='% 4.4f' )        

    return bounds
    
# -----------------------------------------------------------------------------
def xyz2bounds(x, y, z, lb, ub, path=mdir, name_bounds='bounds.den' ):
    
    os.makedirs(path, exist_ok=True)
    
    yxzd = xyz_sort( np.column_stack( ( x, y, z, lb, ub ) ) )
    y, x, z, lb, ub = np.hsplit( yxzd , 5 )
    
    bounds = create_bound_den( lb, ub, path=path, name_bounds=name_bounds )

    return bounds

# -----------------------------------------------------------------------------
def create_obsloc(xloc, yloc, zloc, name_obsloc='obs.loc', path=mdir, round_num=2 ):
    
    os.makedirs(path, exist_ok=True)
    
    ndat = np.size(xloc)
    obs = np.column_stack( ( xloc, yloc, zloc ) )
    
    obs_loc = path + s + name_obsloc
    np.savetxt( obs_loc, obs, fmt=f'%.{round_num}f', header=str(ndat), comments='' )

    return obs_loc

# -----------------------------------------------------------------------------
def create_topodat( xobs, yobs, zobs, path=mdir, name_topodat='topo.dat', round_num=2 ):
    
    os.makedirs(path, exist_ok=True)

    ndat = np.size(xobs)
          
    obs = np.column_stack( ( xobs, yobs, zobs ) )
    
    obs_grv = path + s + name_topodat
    np.savetxt( obs_grv, obs, fmt=f'%.{round_num}f', header=str(ndat), comments='' )

    return obs_grv

# -----------------------------------------------------------------------------
def create_obsgrv( xobs, yobs, zobs, gobs, err=0.05, path=mdir, 
                   name_obsgrv='obs.grv', round_num=2, remove_mean=True ):
    
    os.makedirs(path, exist_ok=True)

    ndat = np.size(xobs)
    
    if remove_mean is True :
        gobs = copy.copy( gobs )
        print( 'g_mean: ', np.nanmean( gobs ) )
        gobs = gobs - np.nanmean( gobs )
    
    if type( err ) in ( int, float ) :
        err = np.repeat( err, ndat )
        
    nan = np.isnan( gobs )
    
    if np.any( nan ) :
        xobs, yobs, zobs, gobs, err =  xobs[~nan], yobs[~nan], zobs[~nan], gobs[~nan], err[~nan]
        ndat = np.size(xobs)
          
    obs = np.column_stack( ( xobs, yobs, zobs, gobs, err ) )
    
    obs_grv = path + s + name_obsgrv
    np.savetxt( obs_grv, obs, fmt=f'%.{round_num}f', header=str(ndat), comments='' )

    return obs_grv

# -----------------------------------------------------------------------------
def gzfor3d( mesh, obs_loc, model_den, path=mdir, topo_dat='', out_file='gzfor3d.grv', runjob=True ):
    
    os.makedirs(path, exist_ok=True)
    
    gzfor3d_exe = mdir + s + 'gzfor3d.exe' 
    
    if path != mdir:
        new_gzfor3d_exe = path + s + 'gzfor3d.exe' 
        if os.path.isfile(new_gzfor3d_exe) is False:
            shutil.copy2( gzfor3d_exe, new_gzfor3d_exe )
            # os.system(f'{pre} copy {gzfor3d_exe} {new_gzfor3d_exe}')
    
    gz_for_3d = path + s + out_file
    
    if shutil._samefile( mesh, path+s+mesh.split(s)[-1] ) is False : 
        shutil.copy2( mesh, path+s+mesh.split(s)[-1] )
    mesh = mesh.split(s)[-1]
    if shutil._samefile( obs_loc, path+s+obs_loc.split(s)[-1] ) is False :
        shutil.copy2( obs_loc, path+s+obs_loc.split(s)[-1] )
    obs_loc = obs_loc.split(s)[-1]
    if topo_dat != '' :
        if shutil._samefile( topo_dat, path+s+topo_dat.split(s)[-1] ) is False :
            shutil.copy2( topo_dat, path+s+topo_dat.split(s)[-1] )
        topo_dat = topo_dat.split(s)[-1]
    if type( model_den ) not in ( float, int ) :
        if shutil._samefile( model_den, path+s+model_den.split(s)[-1] ) is False :
            shutil.copy2( model_den, path+s+model_den.split(s)[-1] )
        model_den = model_den.split(s)[-1]
    
    home_dir = os.getcwd()
    os.chdir(path)

    if runjob == True :
        run = f'{pre} gzfor3d {mesh} {obs_loc} {model_den} {topo_dat}'
        print(run)
        os.system( run )
    g_for = np.loadtxt( out_file, skiprows=1, usecols=[-1] ) 
    
    os.chdir(home_dir)

    return g_for, gz_for_3d

# -----------------------------------------------------------------------------
def gzsen3d( mesh, obs_grv, topo_dat='null', iwt=1, beta='null', znot='null', 
             wvlet='daub2', itol='null', eps='null', path=mdir, out_file='gzinv3d.mtx',
             runjob=True ):
    
    os.makedirs(path, exist_ok=True)
    
    gzsen3d_exe = mdir + s + 'gzsen3d.exe' 
    
    if path != mdir:
        new_gzsen3d_exe = path + s + 'gzsen3d.exe' 
        if os.path.isfile(new_gzsen3d_exe) is False:
            shutil.copy2( gzsen3d_exe, new_gzsen3d_exe )
            # os.system(f'{pre} copy {gzsen3d_exe} {new_gzsen3d_exe}')
    
    gzinv3d_mtx = path + s + out_file
    
    mesh = mesh.split(s)[-1]
    obs_grv = obs_grv.split(s)[-1]
    if topo_dat != 'null' :
        topo_dat = topo_dat.split(s)[-1]
    
    with open(path+s+'gzsen3d.inp', 'w') as fw : 
        fw.write(mesh +'\n')
        fw.write(obs_grv +'\n')
        fw.write(topo_dat +'\n')
        fw.write(str(iwt) +'\n')
        fw.write(str(beta)+' '+str(znot) +'\n')
        fw.write(wvlet +'\n')
        fw.write(str(itol)+' '+str(eps) +'\n')
        fw.close()

    home_dir = os.getcwd()
    os.chdir(path)

    if runjob == True :
        run = f'{pre} gzsen3d.exe gzsen3d.inp'
        os.system( run )
        
    os.chdir(home_dir)

    return gzinv3d_mtx

# -----------------------------------------------------------------------------
def gzinv3d( mesh, obs_grv, ini_den='null', ref_den=0.0, bounds_den=[-2, 2], 
             irest=0, mode=1, par=1, tolc=0.02, L='null', w_dat='null', topo_dat='null', 
             iwt=2, beta='null', znot='null', wvlet='daub2', itol='null', eps='null',
             idisk=0, den0=None, path=mdir, out_file='final_den.den', mask_den=None,
             m_open=False, sen=False, runjob=True ):
    
    os.makedirs(path, exist_ok=True)
    
    gzinv3d_exe = mdir + s + 'gzinv3d.exe' 
    
    if path != mdir:
        new_gzinv3d_exe = path + s + 'gzinv3d.exe' 
        if os.path.isfile(new_gzinv3d_exe) is False:
            shutil.copy2( gzinv3d_exe, new_gzinv3d_exe )
            # os.system(f'copy {gzinv3d_exe} {new_gzinv3d_exe}')
    
    if shutil._samefile( mesh, path+s+mesh.split(s)[-1] ) is False : 
        shutil.copy2( mesh, path+s+mesh.split(s)[-1] )
    # os.system(f'{pre} copy {mesh} {path+s+mesh.split(s)[-1]}')
    mesh = mesh.split(s)[-1]
    if shutil._samefile( obs_grv, path+s+obs_grv.split(s)[-1] ) is False :
        shutil.copy2( obs_grv, path+s+obs_grv.split(s)[-1] )
    # os.system(f'{pre} copy {obs_grv} {path+s+obs_grv.split(s)[-1]}')
    obs_grv = obs_grv.split(s)[-1]
    if topo_dat != 'null' :
        if shutil._samefile( topo_dat, path+s+topo_dat.split(s)[-1] ) is False :
            shutil.copy2( topo_dat, path+s+topo_dat.split(s)[-1] )
        # os.system(f'{pre} copy {topo_dat} {path+s+topo_dat.split(s)[-1]}')
        topo_dat = topo_dat.split(s)[-1]
    if ini_den != 'null' :
        if shutil._samefile( ini_den, path+s+ini_den.split(s)[-1] ) is False :
            shutil.copy2( ini_den, path+s+ini_den.split(s)[-1] )
        # os.system(f'{pre} copy {ini_den} {path+s+ini_den.split(s)[-1]}')
        ini_den = ini_den.split(s)[-1]
    if type( ref_den ) not in ( float, int ) :
        if shutil._samefile( ref_den, path+s+ref_den.split(s)[-1] ) is False :
            shutil.copy2( ref_den, path+s+ref_den.split(s)[-1] )
        # os.system(f'{pre} copy {ref_den} {path+s+ref_den.split(s)[-1]}')
        ref_den = ref_den.split(s)[-1]   
    if type( bounds_den ) in ( tuple, list ) : 
        # shutil.copy2( bounds_den, path+s+bounds_den.spli )
        bounds_den = f'{bounds_den[0]} {bounds_den[1]}'
    elif type( bounds_den ) == str :
        if shutil._samefile( bounds_den, path+s+bounds_den.split(s)[-1] ) is False :
            shutil.copy2( bounds_den, path+s+bounds_den.split(s)[-1] )
        # os.system(f'{pre} copy {bounds_den} {path+s+bounds_den.split(s)[-1]}')
        bounds_den = bounds_den.split(s)[-1]
        
    if sen == True : 
        gzinv3d_mtx = gzsen3d( mesh, obs_grv, topo_dat=topo_dat, iwt=iwt, beta=beta, znot=znot, 
                               wvlet=wvlet, itol=itol, eps=eps, path=path, runjob=True )
        gzinv3d_mtx = gzinv3d_mtx.split(s)[-1]
    else :
        gzinv3d_mtx = 'gzinv3d.mtx'
    
    with open(path+s+'gzinv3d.inp', 'w') as fw : 
        fw.write(str(irest)+'\n')
        fw.write(str(mode)+'\n')
        fw.write(str(par)+' '+str(par)+'\n')
        fw.write(str(obs_grv)+'\n')
        fw.write(str(gzinv3d_mtx)+'\n')
        fw.write(str(ini_den)+'\n')
        fw.write(str(ref_den)+'\n')
        fw.write(str(bounds_den)+'\n')
        fw.write(str(L)+'\n')
        fw.write(str(w_dat)+'\n')
        fw.write(str(idisk)+'\n')
        fw.close()    

    home_dir = os.getcwd()
    os.chdir(path)
        
    if runjob == True :    
        run = f'{pre} gzinv3d.exe gzinv3d.inp'
        print(run)
        os.system( run )

    inv_den = np.loadtxt( 'gzinv3d.den' )
    np.savetxt( out_file, inv_den )

    if den0 != None :
        inv_den[ inv_den != -100.0 ] += den0
        np.savetxt( out_file, inv_den, fmt='% 4.4f')
    if mask_den is not None :
        inv_den[ mask_den.ravel() ] = -100.0
        np.savetxt( out_file, inv_den, fmt='% 4.4f')

    if m_open is True :
        if os.path.isfile( out_file ) :
            meshtools3d( mesh, out_file )
        else :
            if os.path.isfile( 'gzinv3d.den' ) :
                meshtools3d( mesh, 'gzinv3d.den' )
            
    g_pre = np.loadtxt( 'gzinv3d.pre', skiprows=1, usecols=[-1] ) 
    g_obs = np.loadtxt( obs_grv, skiprows=1, usecols=[-2] ) 
        
    os.chdir(home_dir)
    
    density = path + s + out_file
    
    grav = np.column_stack( ( g_obs, g_pre, g_obs - g_pre ) )

    return grav, density        

# -----------------------------------------------------------------------------
def mesh2xyz( mesh, plot=False ) :
    
    with open( mesh, 'r' ) as f :
        lines = f.readlines()
        lines = [line.split() for line in lines]
        
    for i, l in enumerate( lines ) :
        lines[i] = [ float( v ) for v in l ]
        
    nx, ny, nz = lines[0][0], lines[0][1], lines[0][2]
    E0, N0, Z0 = lines[1][0], lines[1][1], lines[1][2]
    # print(nx, ny, nz)
    # print(E0, N0, Z0)
    
    x, y, z = [E0], [N0], [Z0]
    xn, yn, zn = E0, N0, Z0

    for i in lines[2] :
        xn += i 
        x.append( xn )
        
    for i in lines[3] :
        yn += i 
        y.append( yn )
        
    for i in lines[4] :
        zn -= i 
        z.append( zn )
        
    x = np.array( x )
    y = np.array( y )
    z = np.array( z )
    
    xn = np.zeros( x.shape[0]-1 )
    yn = np.zeros( y.shape[0]-1 )
    zn = np.zeros( z.shape[0]-1 )
    
    for i, v in enumerate( x ) :
        if i == 0 : 
            continue
        xn[i-1] = x[i-1] + ( x[i] - x[i-1] ) / 2
    for i, v in enumerate( y ) :
        if i == 0 : 
            continue
        yn[i-1] = y[i-1] + ( y[i] - y[i-1] ) / 2
    for i, v in enumerate( z ) :
        if i == 0 : 
            continue
        zn[i-1] = z[i-1] + ( z[i] - z[i-1] ) / 2
        
    X, Y, Z = np.meshgrid( xn, yn, zn )
    
    if plot == True :
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter( X, Y, Z )
    
    return X, Y, Z

# -----------------------------------------------------------------------------        
def model_den2xyz( model_den, mesh, plot=False, nans=False, nan_val=-100 ) :
    
    d = np.loadtxt( model_den )
    X, Y, Z = mesh2xyz( mesh, plot=False )
    D = d.reshape( Z.shape )
    D[ D == nan_val ] = np.nan
    
    return [ X, Y, Z, D ] 

# -----------------------------------------------------------------------------
def px2nodes( x, y, z, plot=False, plot_level=0 ) :

    zu = -np.sort( -np.unique( z ) )
    n_layer = zu.size
    nodes_x = np.empty( (0,) )
    nodes_y = np.empty( (0,) )
    nodes_z = np.empty( (0,) )
    z_new = zu[:-1] + np.diff( zu )/2  
    new_z0 = z_new[0] - np.diff( zu )[0]
    new_z1 = z_new[-1] + np.diff( zu )[-1]
    z_new = np.insert(z_new, 0, new_z0, axis=0)
    z_new = np.append(z_new, new_z1 )

    for l in range( n_layer ) :
        
        xu = np.unique( x[ z == zu[ l ] ] )
        yu = np.unique( y[ z == zu[ l ] ] )
        Xm,Ym = np.meshgrid( xu, yu )

        Xm1 = Xm[:-1,:-1] + np.diff( Xm )[:-1,:]/2 
        Ym1 = Ym[:-1,:-1] + np.diff( Ym[:,:], axis=0 )[:,:-1]/2    
        
        new_x0 = Xm1[:,0] - np.diff( Xm )[:-1,0]
        new_x1 = Xm1[:,-1] + np.diff( Xm )[:-1,-1]
        Xm1 = np.column_stack( ( new_x0, Xm1, new_x1 ) )
        Xm1 = np.vstack( ( Xm1[0,:], Xm1, Xm1[-1,:] ) )
        
        new_y0 = Ym1[0,:] - np.diff( Ym, axis=0  )[0,:-1]
        new_y1 = Ym1[-1,:] + np.diff( Ym, axis=0  )[-1,:-1]
        Ym1 = np.vstack( ( new_y0, Ym1, new_y1 ) )
        Ym1 = np.column_stack( ( Ym1[:,0], Ym1, Ym1[:,-1] ) )
        
        Zm = np.full( Xm1.shape, z_new[ l ] )
        
        nodes_x = np.concatenate( ( nodes_x, Xm1.ravel() ), axis=0 )
        nodes_y = np.concatenate( ( nodes_y, Ym1.ravel() ), axis=0 )
        nodes_z = np.concatenate( ( nodes_z, Zm.ravel() ), axis=0 )
        
    Zm = np.full( Xm1.shape, z_new[ -1 ] )
    nodes_x = np.concatenate( ( nodes_x, Xm1.ravel() ), axis=0 )
    nodes_y = np.concatenate( ( nodes_y, Ym1.ravel() ), axis=0 )
    nodes_z = np.concatenate( ( nodes_z, Zm.ravel() ), axis=0 )
    
    if plot is True :
        
        plt.subplot(2,1,1)
        xup = np.unique( nodes_x[ nodes_z == z_new[ plot_level ] ] )
        yup = np.unique( nodes_y[ nodes_z == z_new[ plot_level ] ] )
        Xe, Ye = np.meshgrid( xup, yup )        
        plt.pcolor( Xe, Ye, np.ones( (Xm.shape[0], Xm.shape[1]) ), 
                    edgecolors='black', cmap='Blues' )
        xu = np.unique( x[ z == zu[ plot_level ] ] )
        yu = np.unique( y[ z == zu[ plot_level ] ] )
        Xc,Yc = np.meshgrid( xu, yu )    
        plt.scatter( Xc, Yc, c='r' )
        plt.yticks( yu )
        plt.xticks( xu )
        plt.title('Horizzontal_spacing')
        
        plt.subplot(2,1,2 )
        znu = np.unique( nodes_z )
        zu = np.unique( z )
        zux = np.full( zu.shape, (nodes_x.max()-nodes_x.min())/2+nodes_x.min() )
        for l in znu : 
            plt.hlines( l, nodes_x.min(), nodes_x.max(), colors='k' )  
        plt.scatter( zux, zu, c='r' )
        plt.yticks( zu )
        plt.xticks( xu )        
        plt.title('Vertical_spacing')
        
        plt.tight_layout()
        
    return nodes_x, nodes_y, nodes_z





