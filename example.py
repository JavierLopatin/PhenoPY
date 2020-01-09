import pandas as pd
import phenopy as phen
#import matplotlib.pyplot as plt

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

import rasterio

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
PhenoShape(inData=inData, outData=outData, dates=dates, rollWindow=5,
           nan_replace=-32767, nGS=46, chuckSize=16, n_jobs=3)

# get land surface phenology metrics
PhenoLSP(inData=outData, outData=outData[:-4]+'_LSP2.tif', doy=doy, phentype=2,
         nGS=46, n_jobs=8, chuckSize=16)

# get RMSE between the fitted phenoshape and the real distribution of values
RMSE(inData, outData, outData[:-4]+'_RMSE.tif', dates)



PhenoLSP(inData='/home/javier/Documents/SF_delta/Sentinel/LSP/X-004_Y-001_phenoshape.tif',
         outData='/home/javier/Documents/SF_delta/Sentinel/LSP/X-004_Y-001_LSP2.tif', doy=doy, phentype=1,
         nGS=46, n_jobs=8, chuckSize=128)
