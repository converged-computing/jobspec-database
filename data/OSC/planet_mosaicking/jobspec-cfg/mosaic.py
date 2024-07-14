#!/usr/bin/env python3

import sys
import glob
import pickle
from mosaicDEMTiles import mosaicDEMTiles
if sys.version_info[0] < 3:
	import raster_array_tools as rat
else:
	from lib import raster_array_tools as rat

tileDir = '/fs/ess/PZS0720/skhuvis/mosaicking/Iturralde/tiles'
fileNames = sorted(glob.glob(tileDir + '/*0.tif'))
[x,y,z] = mosaicDEMTiles(fileNames)

# Dump x, y, z to pcl file
#with open('mosaic.pcl', 'wb') as f:
#	pickle.dump({'x': x, 'y': y, 'z': z}, f)

# Write to tiff file
geo_trans = rat.extractRasterData(fileNames[0], 'geo_trans')
proj_ref = rat.extractRasterData(fileNames[0], 'proj_ref')
rat.saveArrayAsTiff(z, 'mosaic.tiff', X=x, Y=y, proj_ref=proj_ref, geotrans_rot_tup=(geo_trans[2], geo_trans[4]))
