# In[0]: 
# File Headers
cell_sep = "\n# -----------------------------------------------------------------------------\n"
print( cell_sep + "Cell 0: File header" )

file_headers ="""
\tCreated on 19_07_2024

\t@author: Zampa Luigi Sante
\t@email_1: lzampa@ogs.it
\t@email_2: zampaluigis@gmail.com
\t@org: National Institute of Oceanography and Applied Geophysics - OGS
"""

print( file_headers )

# -----------------------------------------------------------------------------
# In[1]: 
# Import base modules
print( cell_sep + "Cell 1: Import base modules")

# Import base modules
import os
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Print base modules
recording = False
with open(__file__, 'r') as file:
     for line in file:
          if "Import base" in line:
              recording = True; continue
          if "Print base" in line: break 
          if recording: print('\t' + line.strip())
file.close()

# Set "s" variable as system separator
s = os.sep
# Set "rld" variable as importlib.reload
rld = importlib.reload
# Print variables "s" and "rld"
print( '\ts = os.sep\n\trld = importlib.reload' )

# -----------------------------------------------------------------------------
# In[2]: 
# Import local modules
print( cell_sep + "Cell 2: Import local modules")

# Import local modules
import lszpy.utils as utl
import lszpy.gravmag_processing as gmp
import lszpy.plot as plot
import lszpy.gravmag_analysis as gma

# Print local modules
recording = False
with open(__file__, 'r') as file:
     for line in file:
          if "Import local" in line:
              recording = True; continue
          if "Print local" in line: break 
          if recording: print('\t' + line.strip())
file.close()

# -----------------------------------------------------------------------------
# In[3]: 
# Set local dir. paths
print( cell_sep + "Cell 3: Set local dir. paths\n")

# Get current file name and path 
filename = os.path.basename( __file__ ).split('.')[0]
filepath = os.path.abspath( __file__ )

# Set project home dir.
hdir = os.sep.join( filepath.split( os.sep )[:-1] )

# Create local paths dictionary
p = {
     "home" :    hdir,
     "shp" :     hdir+s+'gis'+s+'shp',
     "raster" :  hdir+s+'gis'+s+'raster',
     "py" :      hdir+s+'python',
     "fig" :     hdir+s+'figures',
     "data":     hdir+s+'data',
     }

# Print path dictionary 
print( '\tp = {' )
for key, value in sorted(p.items(), key=lambda x: x[0]):
    print("\t'{}' : '{}'".format(key, value))
print( '\t}' )


# -----------------------------------------------------------------------------
#BlaBlaBla
# %%
