import pandas as pd
import phenopy as phen
import matplotlib.pyplot as plt

days = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/dates.txt'
inData = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/2015-2019_001-365_LEVEL4_TSA_SEN2L_EVI_C0_S0_FAVG_TY_C95T_TSS.tif'

dates = pd.read_csv(days, header=None)[0]
dates = pd.to_datetime(dates)
doy = dates.dt.dayofyear
doy

# Plot
X = 584543.89281
Y = 4228482.52520
phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, type='linear', plotType=1,
               rollWindow=7, ylab='EVI')

phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, type='linear', plotType=2,
               rollWindow=7, ylab='EVI')

# produce shape of phenology
inData = '/home/javier/Documents/SF_delta/Sentinel/npphen_consulta/rasterTest.tif'
outData = '/home/javier/Documents/SF_delta/Sentinel/npphen_consulta/outShape.tif'
phen.PhenoShape(inData=inData, outData=outData, dates=dates, interpolType='linear',
                rollWindow=5, nan_replace=-32767, nGS=46, chuckSize=256, n_jobs=4)

# get land surface phenology metrics
phen.PhenoLSP(inData=outData, outData=outData[:-4]+'_LSP.tif', doy=doy,
              nGS=46, n_jobs=4, chuckSize=256)
