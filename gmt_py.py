# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:20:36 2019

@author: lzampa
"""

import os
import numpy as np
import re
import platform
from . import utils as utl

#==============================================================================
#==============================================================================
#==============================================================================    

class points():
    
    def __init__(self, X, Y, Z, fig=None, geotif=None, xyz=None, rx=None,
                 ry=None, stat=None, ZZ=None, prjcode_in=4326, prjcode_out=4326):

        if prjcode_in != prjcode_out:
            X, Y = utl.prjxy(prjcode_in, prjcode_out, X, Y)
        
        self.X = X.ravel()
        self.Y = Y.ravel()
        self.Z = Z.ravel()
        
        nan = np.isnan( self.Z )
        self.Z = self.Z[ ~nan ] 
        self.X = self.X[ ~nan ] 
        self.Y = self.Y[ ~nan ] 
        
        self.fig = fig 
        self.geotif = geotif
        self.xyz = xyz
        self.ZZ = ZZ
        self.rx = rx
        self.ry = ry
        self.stat = stat       
        
    
#%% ===========================================================================
### Python function
### ===========================================================================

    def gmtplt_surf(self, 
                    lim1=None, lim2=None, lim3=None, 
                    vmin=None, vmax=None, vstep='',
                    blkm=None, gstep=None, wfilt=None, 
                    cstep='auto', lcstep=None, 
                    lcsize='9p',
                    N=1,
                    map_proj='-JM16c', 
                    imap_dim='4c', imap_x='0.1c', imap_y='0.1c', 
                    Azim=300, 
                    r_msk=None, 
                    T=0.3,
                    clscale='wysiwyg', 
                    psscale='-DJMB+h+e -Baf -Bx+lunits',
                    coast_properties='-Da -A+ag -W0.5p',
                    land_cover=False, land_cover_color='205/183/158',
                    points=False, dim_points='+0.05c',
                    add_points=[], points_properties='-S+0.05c -Gblack',
                    add_rectangle=[], rectangle_properties='-W2p,red',
                    shp_line=None, pr_shp_line='1p,black,-',
                    map_scale=False, 
                    map_scale_parameters='-F+p+gwhite -LjBL+c%xmin1%/%ymin1%+w20k+l+o1c/1c',
                    MAP_FRAME_AXES='WNes',
                    title=None, 
                    text=None,
                    roundnum=5,
                    h_pad=0.2,
                    v_pad=0.2,
                    pstext='-F+cMC',
                    name_map='out_map', path=None,
                    im_format='jpg', im_res='', 
                    pause=False, open_fig=True,
                    FONT_TITLE ='14p,Helvetica', 
                    FONT_ANNOT_PRIMARY='14p,Helvetica-Oblique', 
                    FONT_LABEL='14p,Helvetica',
                    MAP_FRAME_TYPE='fancy'):       

# -----------------------------------------------------------------------------
### Self
        X = self.X
        Y = self.Y
        Z = self.Z

# -----------------------------------------------------------------------------
### Create folder with output 
        H_dir = os.getcwd()
        if path==None:    
            gmt_p = H_dir + os.sep + name_map
        else:
            gmt_p = path + os.sep + name_map
    
        os.makedirs(gmt_p, exist_ok=True)
        os.chdir(gmt_p)
        import glob
        files = glob.glob(gmt_p+os.sep+'*')
        for f in files:
            os.remove(f)
# -----------------------------------------------------------------------------
### If Min_X,Y,Z & Max_X,Y,Z is None, then calc. missing Min Max from X,Y,Z input data 

        
        if  lim2 is None:
            MinX2 = round(np.min(X),8)
            MaxX2 = round(np.max(X),8)
            MinY2 = round(np.min(Y),8)
            MaxY2 = round(np.max(Y),8)
        else:   
            MinX2 = lim2[0]
            MaxX2 = lim2[1]
            MinY2 = lim2[2]
            MaxY2 = lim2[3]            

        if  lim1 is None:
            MinX1 = MinX2
            MaxX1 = MaxX2
            MinY1 = MinY2
            MaxY1 = MaxY2            
        else:   
            MinX1 = lim1[0]
            MaxX1 = lim1[1]
            MinY1 = lim1[2]
            MaxY1 = lim1[3]
            
        if vmin == None: vmin = round(np.nanmean(Z)-2*np.nanstd(Z),roundnum)
        else: vmin = vmin;
            
        if vmax == None: vmax = round(np.nanmean(Z)+2*np.nanstd(Z),roundnum)
        else: vmax = vmax; 
        
        if vstep == None : vstep=''
        else: vstep = '/'+str( vstep )
# -----------------------------------------------------------------------------
### Set parameters of iMap (small mapframe indicating geographic position of the study area) -->lim3
        
        if  lim3 is None:
            set_imap = ''
        else:    
            MinX3 = lim3[0]
            MaxX3 = lim3[1]
            MinY3 = lim3[2]
            MaxY3 = lim3[3]
            set_imap = f'''
:: lim3 = LIMITI della MAPPA piccola di INQUADRAMENTO 
:: lim3 = -Rlongitudine_minima/longitudine_massima/latitudine_minima/latitudine_massima (NB. non cancellare +r alla fine)
:: imap_dim = larghezza orizzontle della MAPPA piccola di INQUADRAMENTO (N.B ricordati di mettere la c di cm dopo il numero)
:: imap_X, imap_Y = posizione X,Y della MAPPA piccola di INQUADRAMENTO rispetto all'angolo in basso a sinistra della cornice
set xmin3={MinX3}
set xmax3={MaxX3}
set ymin3={MinY3}
set ymax3={MaxY3}
:: Non modificare -------------------------------------------------------------      
set lim3=-R%xmin3%/%ymin3%/%xmax3%/%ymax3%+r
set lim3=%lim3: =%
::----------------------------------------------------------------------------- 
set imap_dim={imap_dim}
set imap_Y={imap_x}
set imap_X={imap_y}'''
       
### ---------------------------------------------------------------------------             
### Calculate half avarege minimum distance between scatterd XYZ points (this will be the default grid step)
            
        n_points=np.array([self.X,self.Y]).T
        if gstep == None:
            md = []
            half_average_min = []
            for i in range(len(n_points)):
                apoints=[]
                apoints=np.delete(n_points,i,0)
                d = np.sqrt(((n_points[i,0]-apoints[:,0])**2)+((n_points[i,1]-apoints[:,1])**2));
                md += [np.min(d)]
            
            half_average_min=np.mean(md)/2                        
            gstep = round(half_average_min, 10);

### ---------------------------------------------------------------------------
### Set block mean filter (anti alias)

        if blkm == None:
            if type(gstep) is str: 
                mre = re.compile(r'([\d*\.?\d+]+)([a-zA-Z]+)')
                mmre = mre.match(gstep)
                blkm = f'{float(mmre.group(1))}{mmre.group(2)}'
            else:
                blkm = gstep

### ---------------------------------------------------------------------------
### Set add_points (plot additiona points)

        if add_points != []:
            if type(add_points) in (list,tuple) : 
                add_points = np.column_stack( ( add_points[0], add_points[1] ) )
            np.savetxt('add_points', add_points, fmt='%f')      
            
### ---------------------------------------------------------------------------
### Set shp_lines
            
        if shp_line != None:
            os.system(f'ogr2ogr -f "OGR_GMT" "shp_line.gmt" "{shp_line}"')            
            
### ---------------------------------------------------------------------------
### Set default countur line interval (40 lines dividing the Min-Max Z_range)           
        if cstep!=None:
            if cstep == 'auto': 
                cstep = np.round((vmax-vmin)/10,roundnum)
            if lcstep == None:
                if type(cstep) is str:             
                    mre = re.compile(r'([\d*\.?\d+]+)([a-zA-Z]+)')
                    mmre = mre.match(cstep)
                    lcstep = f'{float(mmre.group(1))*2}{mmre.group(2)}+f8p'
                else: lcstep = cstep*2            
            
### ---------------------------------------------------------------------------
### Set default gaussian filter window        
        
        if wfilt == None:
                wfilt = 3
                
### ---------------------------------------------------------------------------
### Set radius of grdmask 
            
        if r_msk is not None:
            set_r_msk=f'''
:: r_msk = raggio oltre il quale viene applicata la maschera a partire dai punti originali (es. -M10k = a partire da 10 km dal punto)
set r_msk={r_msk}'''
        else:
            set_r_msk=''

### ---------------------------------------------------------------------------
### Set land areas default color, (e.g. when plotting only off-shore data) 
            
        pcover =''
        if land_cover == True:pcover = f'-G{land_cover_color}'
        else: pcover = pcover
              
### ---------------------------------------------------------------------------
### Set title 
        
        _,_,idx = utl.xy_in_lim( self.X, self.Y, [MinX2,MaxX2,MinY2,MaxY2] )   
        
        if title!=None:         
            if type(title)==int:
                mmms = (np.min(Z[idx]),np.max(Z[idx]),np.mean(Z[idx]),np.std(Z[idx]))
                title = (f'[ Min={np.round(mmms[0],title)}  Max={np.round(mmms[1],title)}  Mean={np.round(mmms[2],title)}  Std={np.round(mmms[3],title)} ]')
            Title = f'-B+t"{title}"'
        else: Title = ''  

### ---------------------------------------------------------------------------
### Set text 
        
        stat = np.nanmin(Z), np.nanmax(Z), np.nanmean(Z), np.nanstd(Z), np.sqrt(np.nanmean(Z**2))
        Min,Max,Mean,Std,Rms = stat
        if text!=None:         
            if type(text)==int:
                text = f'[ Min={Min:.{text}f}  Max={Max:.{text}f}  Mean={Mean:.{text}f}  Std={Std:.{text}f} ]'
            with open ('pstext.d', 'w') as tw: tw.write(text)
     
### ---------------------------------------------------------------------------
### Set output image format for gmt conversion (default  = jpg = -Tj) 
               
        if open_fig == True: 
            open_map = f'%name_map%.{im_format}'
        else: open_map=''  
        if im_format == 'bmp': im_format_gmt = '-Tb'
        if im_format == 'png': im_format_gmt = '-Tg'
        if im_format == 'jpg': im_format_gmt = '-Tj'
        if im_format == 'tif': im_format_gmt = '-Tt'
        if im_format == 'svg': im_format_gmt = '-Ts'
        if im_format == 'eps': im_format_gmt = '-Te'
        if im_format == 'pdf': im_format_gmt = '-Tf'

### ---------------------------------------------------------------------------
### Set pause
             
        if pause == True: ps = 'pause'                      
        else: ps=''
            
### ---------------------------------------------------------------------------
### Create temporary file with XYZ data to be used as input in gmt functions 
               
        temp_xyz = np.column_stack((self.X,self.Y,self.Z)) 
        np.savetxt('temp', temp_xyz, fmt='%f')

### ---------------------------------------------------------------------------
### ===========================================================================          
### GMT input var 
### ===========================================================================               
        gmt_input_p =f'''
::=============================================================================
:: Parametri da impostare =====================================================

:: name_map = nome della mappa e delle figure che otterai come file di output (es. output.jpg) 
:: NB. scivi SOLO il nome NON AGGIUNGERE .formato !!! importante, se no fa casino
set name_map={name_map}

:: name_file = nome file in ingresso con i punti xyz da interpolare
:: NB. scivi SOLO il nome NON AGGIUNGERE .formato MA il formpato deve avere estensione .txt
set name_file=temp

:: lim1 = LIMITI della CORNICE ESTERNA
:: lim1 = -Rlongitudine_minima/longitudine_massima/latitudine_minima/latitudine_massima (NB. non cancellare +r alla fine)
set xmin1={MinX1}
set xmax1={MaxX1}
set ymin1={MinY1}
set ymax1={MaxY1} 
:: Non modificare -------------------------------------------------------------         
set lim1=-R%xmin1%/%ymin1%/%xmax1%/%ymax1%+r
set lim1=%lim1: =%
::-----------------------------------------------------------------------------   
:: lim2 = LIMITI della SUPERFICIE da interpolare INTERNA alla mappa (deve essere + piccola della cornice esterna)
:: lim2 = -Rlongitudine_minima/longitudine_massima/latitudine_minima/latitudine_massima (NB. non cancellare +r alla fine)            
set xmin2={MinX2}
set xmax2={MaxX2}
set ymin2={MinY2}
set ymax2={MaxY2} 
:: Non modificare -------------------------------------------------------------   
set lim2=-R%xmin2%/%ymin2%/%xmax2%/%ymax2%+r
set lim2=%lim2: =%
::-----------------------------------------------------------------------------
   
{set_imap}

:: map_proj = proiezione utilizzata per la mappa (es -JM16c = priezione di mercatore, mappa larga 16 cm) 
set map_proj={map_proj}

:: Ampiezza della finestra per la media/mediana a blocchi
set blkm={blkm}

:: Passo di campionamento della griglia (unit=k km)
set gstep={gstep}

:: Filtro gaussiano passa basso (k = km) --> smussa la superficie togliendo le alte frequenze
set wfilt={wfilt}

:: dir_luce = Azimuth del sole = direzione angolare da cui proviene la luce che crea l'ombra 
:: (dovrebbe essere in gradi decimali da 0 a 360, partendo da Nord e procedendo in senso orario) 
set Azim={Azim} 

{set_r_msk}

:: int_isol = intervallo delle isolinee, ovvero ogni quanto disegna una linea (es. -C0.003 --> ogni 0.003 ms)
:: val_isol = etichette con valore della linea (es. -A0.006+fp8 = etichette ogni 0.006 ms con dimensione carattere 8)
set cstep={cstep}
set lcstep={lcstep}

:: Formato coordinate gradi decimali
gmt set FORMAT_GEO_MAP=ddd.mm

:: Formato titolo (12p = dimensione)
gmt set FONT_TITLE={FONT_TITLE}

:: Formato coordinate gradi decimali
gmt set MAP_FRAME_TYPE={MAP_FRAME_TYPE}

:: Disegna assi e annotazioni
gmt set MAP_FRAME_AXES={MAP_FRAME_AXES}

:: Formato annotazioni varie (prova)
gmt set FONT_ANNOT_PRIMARY={FONT_ANNOT_PRIMARY}

:: Formato annotazioni varie (prova)
gmt set FONT_LABEL={FONT_LABEL}

:: Formato scala colore 
:: Tipo di scala (es. jet)
:: alcune scale predefinite le trovi qui 
:: http://lira.epac.to:8080/doc/gmt/html/GMT_Docs.html
:: -T = minimo/massimo (attenzione ai segni!!!)
:: file di output = color.cpt 
::-----------------------------------------------------------------------------
gmt makecpt -C{clscale} -T{vmin}/{vmax}{vstep} -D > color.cpt
::-----------------------------------------------------------------------------'''   
         
### ===========================================================================
### GMT funtions 
### ===========================================================================          
### blockmean 
        blockmean =f'''
:: Applica una media a blocchi (anti aliasing). Voule temp.txt come input --> Restituisce il file temp_bm.xyz come output
::-----------------------------------------------------------------------------
gmt blockmean %name_file% -I%blkm% %lim2% > %name_file%_bm
::-----------------------------------------------------------------------------'''
###----------------------------------------------------------------------------
### surface 
        surface =f'''
:: Crea superficie interpolata dai punti xyz che gli metti in come input (algoritmo minimum curvature modificato)
:: passo di campionamento per gli amici gridstep = -I0.05k (k=kilometri)
:: -G = nome output
::-----------------------------------------------------------------------------
gmt surface %name_file%_bm %lim2% -G%name_map%.nc -I%gstep%+e -T{T}
::-----------------------------------------------------------------------------'''
###----------------------------------------------------------------------------
### filter surface 
        filters =f'''
:: Filtra la superficie interpolata che hai creato nella riga precedente (%name_map%.nc) 
:: -Fg = Filtro gaussiano passa basso che "smussa" tutti i dettagli di dimensione inferiore al parametro %wfilt% (riga 22-23)
:: -G = nome output (sovrascrivitto al precedente)
::-----------------------------------------------------------------------------		
gmt grdfilter %name_map%.nc -Fg%wfilt% -G%name_map%.nc -Dp
::-----------------------------------------------------------------------------'''
###----------------------------------------------------------------------------
### gradient 
        gradient =f'''
:: Crea superficie con i valori del gradiente orizzontale di %name_map%.nc creata nella riga sopra 
:: (serve per ottenere la shadow relief nel comando grdimage, vedi sotto)   
:: -N = fattore di amplificazione verticale (più è grande più "l'effetto 3D" viene esagerato)
:: -A = Azimuth del sole = direzione angolare da cui proviene la luce che crea l'ombra 
:: -Da = luce inversa (eventually)
:: (dovrebbe essere in gradi decimali da 0 a 360, partendo da Nord e procedendo in senso orario)   
:: -G = nome output -- l'ho chiamato in modo diverso rispetto al grid precedente così non si sovrascrive alla superficie originale (_hg sta per horizontal gradient)
::-----------------------------------------------------------------------------
gmt grdgradient %name_map%.nc %lim2% -G%name_map%_hg.nc -N2 -fg -A%Azim% 
::-----------------------------------------------------------------------------'''
###----------------------------------------------------------------------------
### filter gradient 
        filterg = f'''
:: Filtra la superficie del gradiente che hai creato nella riga precedente (%name_map%.nc) 
:: -Fg%filter_window% = Filtro gaussiano passa basso che "smussa" tutti i dettagli di dimensione inferiore al parametro %filter_window% (riga 22-23)
:: -G = output
::-----------------------------------------------------------------------------			   
gmt grdfilter %name_map%_hg.nc -Fg%wfilt% -G%name_map%_hg.nc -Dp
::-----------------------------------------------------------------------------'''
###----------------------------------------------------------------------------
### psbasemap
        basemap = f'''
:: Crea una prima mappa alla base della figura (cornice e grigliato delle coordinate) e il titolo in alto
:: -P = Portrait (foglio in orizzontale, se non lo specifichi lo mette di default in verticale)
:: >> %name_map%.ps = file di output, il formato che usa gmt per le immagini è PostScript (.ps)
::-----------------------------------------------------------------------------
gmt psbasemap %lim1% %map_proj% -Ba {Title} -P -K > %name_map%.ps
::-----------------------------------------------------------------------------'''
###----------------------------------------------------------------------------
### grdimage 
        grdimage = f'''
:: Disegna la superficie creata con la shadow relief all'interno della basemap
:: -I = input del file gradiente per costruire effetto 3D 
:: -C = tipo di colormap
:: >> %name_map%.ps = incolla risulatato sulla mappa creata nel precedente passaggio (sovrascrivendo il file + aggiunta)
::-----------------------------------------------------------------------------							 
gmt grdimage %name_map%.nc -I%name_map%_hg.nc %map_proj% %lim1% -Ccolor.cpt -nb+c -Q -K -O >> %name_map%.ps
::------------------------------------------------------------------------------'''
###---------------------------------------------------------------------------
### grdcontour 
        if cstep!=None: 
            grdcontour = f'''
:: Disegna le isolinee della superficie (se non le vuoi commenta la linea con ::)  
:: -C = intervallo delle isolinee, ovvero ogni quanto disegna una linea (es. -C0.003 --> ogni 0.003 ms)
:: -A = etichette con valore della linea (es. -A0.006+fp8 = etichette ogni 0.006 ms con dimensione carattere 8) 
:: >> %name_map%.ps = incolla risulatato sulla mappa creata nel precedente passaggio (sovrascrivendo il file + aggiunta)
::-----------------------------------------------------------------------------
gmt grdcontour %name_map%.nc %lim1% %map_proj% -C%cstep% -A%lcstep%+f{lcsize} -K -O >> %name_map%.ps
::-----------------------------------------------------------------------------'''
        else: grdcontour=''
###---------------------------------------------------------------------------
### grdmask
        if r_msk!=None: 
            psmask = f'''
:: r_msk = raggio oltre il quale viene applicata la maschera a partire dai punti originali (es. -M10k = a partire da 10 km dal punto)
::-----------------------------------------------------------------------------
gmt psmask %name_file%_bm %lim1% %map_proj% -I%gstep% -S%r_msk% -T -N -Gwhite -K -O >> %name_map%.ps   
::-----------------------------------------------------------------------------'''     
        else: psmask=''           
###----------------------------------------------------------------------------             
### pscoast 
        if map_scale==False: map_scale_parameters=''
        coast =f'''
:: Disegna la linea di costa e la scala della mappa in basso (se non la vuoi commenta la linea con ::)  
:: -E = indica il continente dove stai (-EEU = Europa) 
:: -W = spessore della linea 
:: -F = Disegna un rettagolo bianco come background per la scala della mappa in basso
:: -L = posizione e dimensione della scale della mappa in basso (+w = dimensioni scala in km, +o = offset da xmin1/ymin1)
:: >> %name_map%.ps = incolla risulatato sulla mappa creata nel precedente passaggio (sovrascrivendo il file + aggiunta)
::-----------------------------------------------------------------------------
gmt pscoast %lim1% %map_proj% {coast_properties} {pcover} {map_scale_parameters} -K -O >> %name_map%.ps
::-----------------------------------------------------------------------------'''
###---------------------------------------------------------------------------
### psxy lines
        if shp_line!=None: shp_ln =f'''
:: Disegna linee contenute nel file shp_line.gmt  
::-----------------------------------------------------------------------------
gmt psxy shp_line.gmt %lim1% %map_proj% -W{pr_shp_line} -K -O --MAP_FRAME_TYPE=inside >> %name_map%.ps 
::-----------------------------------------------------------------------------'''
        else: shp_ln='' 
###---------------------------------------------------------------------------            
### psxy gridpoints
        if points==True: points =f'''
:: Disegna i punti originali di calcolo che hai usato come input per creare la superficie  
:: -S = specifiche della forma e dimensione dei punti disegnati (es. -Sc0.05c = cerchi di dimensione 0.05 cm, vedi l'help)
:: -G = specifiche del colore di riempimento (-G00/00/00 = nero)
::-----------------------------------------------------------------------------
gmt psxy %name_file% %lim1% %map_proj% -S{dim_points} -Gblack -K -O --MAP_FRAME_TYPE=inside >> %name_map%.ps 
::-----------------------------------------------------------------------------'''
        else: points=''
###----------------------------------------------------------------------------
### add_psxy points            
        if add_points!=[]: 
            add_p =f'''
:: Disegna in più  
:: -S = specifiche della forma e dimensione dei punti disegnati 
:: (es. -Sc0.05c = cerchi di dimensione 0.05 cm, vedi l'help)
:: -G = specifiche del colore di riempimento (-G00/00/00 = nero)
::-----------------------------------------------------------------------------
gmt psxy add_points %lim1% %map_proj% {points_properties} -K -O --MAP_FRAME_TYPE=inside >> %name_map%.ps 
::-----------------------------------------------------------------------------'''
        else: add_p=''
###----------------------------------------------------------------------------      
### add_rectangle
        if add_rectangle!=[]: add_rectangle=f'''
(                                             
echo {add_rectangle[0]} {add_rectangle[3]}
echo {add_rectangle[1]} {add_rectangle[3]}
echo {add_rectangle[1]} {add_rectangle[2]}
echo {add_rectangle[0]} {add_rectangle[2]}
echo {add_rectangle[0]} {add_rectangle[3]}
)>rectangle.txt
gmt psxy rectangle.txt -R -J {rectangle_properties} -K -O >> %name_map%.ps'''       
        else: add_rectangle=''
###----------------------------------------------------------------------------
### psext
        if text!=None: 
            psext=f'''
gmt pstext pstext.d %lim1% %map_proj% {pstext} -K -O >> %name_map%.ps
::-----------------------------------------------------------------------------'''
        else: psext=''                  
###----------------------------------------------------------------------------         
### psscale 
        if lim3 is not None: K='-K'
        else: K=''             
        cscale =f'''
:: Disegna scala colori (se non la vuoi commenta la linea con ::)
:: -D = posizione della scala colori 
:: (es. -DJMR+e = esterna alla cornice (j minuscola per metterla dentro) 
:: in mezzo a destra(Middle Right, MR) con triangolini in alto e in basso (+e))  
:: -C = input file della scala colori creato nella linea 51
:: -B = formato annotazioni (af non mi ricordo cos'è, +l indica l'unità di misura es. "ms") 
:: -F = Disegna un rettagolo bianco come background per la scala della mappa in basso
:: >> %name_map%.ps = incolla risulatato sulla mappa creata nel precedente passaggio (sovrascrivendo il file + aggiunta)
::-----------------------------------------------------------------------------								   
gmt psscale {psscale} -Ccolor.cpt %lim1% %map_proj% {K} -I -O >> %name_map%.ps
::-----------------------------------------------------------------------------'''
### imap 
        if lim3 is not None: imap =f'''
:: Disegna mappa di inquadramento in alto a sinistra (se non la vuoi commenta le 4 linee con ::)			
:: -J = proiezione (es. -JM=simple mercator) e dimensione orizzontale (es. -JM4c= 4 centimetri)
:: -X = posizione rispetto al bordo sinistro della cornice (e. -X0.01c= 0.01centimetri)
:: -Y = posizione rispetto al bordo sinistro della cornice (e. -Y18c= 18centimetri)
:: >> %name_map%.ps = incolla risulatato sulla mappa creata nel precedente passaggio (sovrascrivendo il file + aggiunta)
::-----------------------------------------------------------------------------
gmtset	MAP_FRAME_TYPE
set dlim3=-D%xmin3%/%ymin3%/%xmax3%/%ymax3%+r
set dlim3=%dlim3: =% 
gmt psbasemap %lim3% -JM%imap_dim% %dlim3% -F -X%imap_x% -Y%imap_y% -K -P -O >> %name_map%.ps
gmt pscoast -R -J -W0.1p,black -A+ag -G240/240/240 -S255/255/255 -Df -K -O >> %name_map%.ps
(
echo %xmin1% %ymax1%
echo %xmax1% %ymax1%
echo %xmax1% %ymin1%
echo %xmin1% %ymin1%
echo %xmin1% %ymax1%
)>map1_area_sqr.txt
gmt psxy map1_area_sqr.txt -R -J -W2p,red -O >> %name_map%.ps
::-----------------------------------------------------------------------------'''
        else: imap=''            
### ---------------------------------------------------------------------------                       
### psconvert
        if im_format_gmt != '-Ts':
            convert =f'''
:: Converti il file .ps in un altro formato immagine più fruibile (es. jpg, png etc..)			
:: -T = formato output (-Tf=pdf, -Tg=png, -Tj=jpg. per gli altri vedi nell'help) 
:: -A = Limita i bordi bianchi che contornano l'immagine -> ovvero margini
::-----------------------------------------------------------------------------
gmt psconvert %name_map%.ps {im_format_gmt} -E{im_res} -Au{h_pad}c/{v_pad}c 
::-----------------------------------------------------------------------------'''
        else:
            convert =f'''
:: Converti il file .ps in un altro formato immagine più fruibile (es. jpg, png etc..)			
:: -T = formato output (-Tf=pdf, -Tg=png, -Tj=jpg. per gli altri vedi nell'help) 
:: -A = Limita i bordi bianchi che contornano l'immagine -> ovvero margini
::-----------------------------------------------------------------------------
gmt psconvert %name_map%.ps -Tf -E{im_res} -Au{h_pad}c/{v_pad}c
pdf2svg %name_map%.pdf %name_map%.svg 
::-----------------------------------------------------------------------------'''        
### ---------------------------------------------------------------------------    
### geotiff_convert 
        geotiff_convert =f'''
gmt psconvert %name_map%.ps -Tt -W+g -E{im_res}        
gdalwarp -overwrite -q -of GTiff -s_srs EPSG:4326 -t_srs EPSG:4326 NETCDF:%name_map%.nc %name_map%.tif
::-----------------------------------------------------------------------------'''
### kml_convert 
        kml_convert =f'''
gdal_translate -of KMLSUPEROVERLAY %name_map%.tiff %name_map%.kmz -co FORMAT=PNG            
::-----------------------------------------------------------------------------'''
### ---------------------------------------------------------------------------    
### grd_convert             
        grd_convert =f'''
:: Convert the nc grid to xyz            
gmt grd2xyz %name_map%.nc > %name_map%.xyz
::-----------------------------------------------------------------------------'''
### ===========================================================================   
### Run code
### ===========================================================================  
        with open ('run.bat', 'w') as rsh:
            rsh.write(f'''
prompt $g                      
::=============================================================================
::=============================================================================
::=============================================================================
::
:: Created on Mon April 2018
:: Author: Luigi Sante Zampa OGS (National Institute of Oceanography and Applied Geophysics)
::
{gmt_input_p}
::=============================================================================
::=============================================================================
::=============================================================================
:: The code start here ========================================================
{blockmean}
{surface}
{filters}
{gradient}
{filterg}
{basemap}
{grdimage}
{grdcontour}
{psmask}
{coast}
{shp_ln}
{points}
{add_p}
{add_rectangle}
{psext}
{cscale}
{imap}
{convert}
{geotiff_convert}
{kml_convert}
{grd_convert}
{ps}
::=============================================================================
::=============================================================================						   
:: Delete unecessary files
del %name_file%_bm
del %name_map%.nc
del %name_map%_hg.nc
del gmt.conf
del color.cpt
del gmt.history
:: del %name_map%.ps

::=============================================================================
::=============================================================================							   
:: Open_map ({open_map})
{open_map}
{ps}
''')
        
        if platform.system() == 'Linux' :
            os.system('wine cmd run.bat')
        else :
            os.system('run.bat')
        
        self.xyz = np.loadtxt(f'{name_map}.xyz')
        x = np.unique(self.xyz[:,0])
        y = np.unique(self.xyz[:,1])
        self.ZZ = np.reshape(self.xyz[:,2], (len(y), len(x)))
        self.ry = np.unique(self.xyz[:,1])[1]-np.unique(self.xyz[:,1])[0]
        self.rx = np.unique(self.xyz[:,0])[1]-np.unique(self.xyz[:,0])[0]
        self.X = np.reshape(self.xyz[:,0], (len(y), len(x)))
        self.Y = np.reshape(self.xyz[:,1], (len(y), len(x)))
        self.XYZ = [ self.X, self.Y, self.ZZ ]
        np.savetxt(f'''{name_map}_grd''', self.ZZ)

### ---------------------------------------------------------------------------        
### Output parameter = name of the image with final map 
        
        os.chdir(H_dir)
        self.fig = gmt_p +os.sep+ name_map + '.' + im_format
        self.geotif = gmt_p +os.sep+ name_map + '.' + 'tiff'
        self.tif = gmt_p +os.sep+ name_map + '.' + 'tif'
        # output = {'fig':path_name_img, 'geotif':path_name_gtiff, 'xyz':xyz, 
        #           'Z':Z, 'rx':rx, 'ry':ry, 'stat':stat}
        
        return self

### ---------------------------------------------------------------------------        
### Method open figure    
    def open_fig(self):
        
        os.system(self.fig)
        print(self.fig)
        
        
### ---------------------------------------------------------------------------  
    


