########################################################################
#
# Process Sentinel-2 time series data to obtain Phenology indices using
# the PhenoPy Library.
#
# Author: Javier Lopatin
#
########################################################################

import rasterio
import pandas as pd
import phenopy as phen

days = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/dates.txt'
inData = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/2015-2019_001-365_LEVEL4_TSA_SEN2L_EVI_C0_S0_FAVG_TY_C95T_TSS.tif'

dates = pd.read_csv(days, header=None)[0]
dates = pd.to_datetime(dates)
doy = dates.dt.dayofyear
doy

# Plot
X = 583515.162
Y = 4231631.138
phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, plotType=1,
               rollWindow=5, ylab='EVI')

phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, plotType=2, phentype=2,
               rollWindow=5, n_phen=10, ylab='EVI')

# produce shape of phenology
inData = '/home/javier/Documents/SF_delta/Sentinel/npphen_consulta/rasterTest.tif'
outData = '/home/javier/Documents/SF_delta/Sentinel/npphen_consulta/outShape.tif'


with rasterio.open(inData) as r:
    print('# colums = ', r.width)
    print('# rows = ', r.height)
    print('# bands = ', r.count)
    dstack = r.read()

with rasterio.open(outData) as r:
    print('# colums = ', r.width)
    print('# rows = ', r.height)
    print('# bands = ', r.count)
    phen = r.read()

# get phenological shape of the wetlands
phen.PhenoShape(inData=inData, outData=outData, dates=dates, rollWindow=5,
                nan_replace=-32767, nGS=46, chuckSize=16, n_jobs=3)

# get land surface phenology metrics
phen.PhenoLSP(inData=outData, outData=outData[:-4] + '_LSP2.tif', doy=doy, phentype=2,
              nGS=46, n_jobs=8, chuckSize=16)

# get RMSE between the fitted phenoshape and the real distribution of values
phen.RMSE(inData, outData, outData[:-4] + '_RMSE.tif', dates)


phen.PhenoLSP(inData='/home/javier/Documents/SF_delta/Sentinel/LSP/X-004_Y-001_phenoshape.tif',
              outData='/home/javier/Documents/SF_delta/Sentinel/LSP/X-004_Y-001_LSP2.tif', doy=doy, phentype=1,
              nGS=46, n_jobs=8, chuckSize=128)

#####################################
# Plot resulting rasters
#####################################

import matplotlib.pyplot as plt
import rioxarray as xr
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd

# rasters
rmse = '/home/javier/Documents/SF_delta/Sentinel/Siusun/RMSE_all.tif'
#rmse2 = '/home/javier/Documents/SF_delta/Sentinel/Siusun/RMSE2.tif'
# load rasters
rmse = xr.open_rasterio(rmse)
#rmse2 = rioxarray.open_rasterio(rmse2)
# set 0 data as NaN
rmse = rmse.where(rmse.values > 1)
# set 0-10,000 scale EVI to 0-1 values
rmse.values = rmse.values / 10000

### plot full Suisun area
rmse.plot(robust=True, cmap=plt.cm.magma, cbar_kwargs={"label": "EVI"})
plt.title('RMSE')
plt.ylabel('')
plt.xlabel('')
scalebar = ScaleBar(10, location='lower left') # 1 pixel = 10 meter
plt.gca().add_artist(scalebar)
plt.tight_layout()
plt.show()

### Zoom to example site
# clipper geometries
geometries = [
    {
        'type': 'Polygon',
        'coordinates': [[
            [579796.887, 4224590.229],
            [583901.700, 4224582.365],
            [583905.632, 4221303.233],
            [579800.818, 4221303.233],
            [579796.887, 4224590.229] # same as first to close the geometry
        ]]
    }
]

clipped = rmse.rio.clip(geometries, rmse.rio.crs)
# plot
clipped.plot(robust=True, cmap=plt.cm.magma, cbar_kwargs={"label": "EVI"})
plt.title('RMSE')
plt.ylabel('')
plt.xlabel('')
scalebar = ScaleBar(10, location='lower left') # 1 pixel = 10 meter
plt.gca().add_artist(scalebar)
plt.tight_layout()
plt.show()

### plot classess in zoom area

shp = gpd.read_file('shapefiles/SuisunMarsh_CalVegName.shp')
# delate barrem, Agriculture and other non-natural classes
shp = shp.iloc[[1,2,4,6,7,8,9,10,11,12,13,14,15,16]] 

# plot
shp.plot(column='CalVegName', cmap='tab20', legend=True)
plt.title('Vegetation types')

# subplot
fig, ax = plt.subplots()
shp.plot(column='CalVegName', ax=ax, cmap='tab20', legend=True)
ax.set_xlim(579796.887, 583901.700)
ax.set_ylim(4221303.233, 4224590.229)
plt.title('Vegetation types')