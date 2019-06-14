import pandas as pd
import phenopy as phen
import matplotlib.pyplot as plt

days = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/dates.txt'
#inData = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-003_Y-001/2015-2019_001-365_LEVEL4_TSA_SEN2L_EVI_C0_S0_FAVG_TY_C95T_TSS.tif'
inData = '/home/javier/Documents/SF_delta/Sentinel/npphen_consulta/rasterTest.tif'
# import dates
dates = pd.read_csv(days, header=None)[0]
dates = pd.to_datetime(dates)
dates
# create dummy set of dates all with same year to mix according to DOY
#years2 = years.apply(lambda dt: dt.replace(year=2018))

# Plot
X = 594389.28100
Y = 4226251.48322
phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, ylim=None, type=1,
    rollWindow=3, ylab='EVI')

phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, ylim=None, type=2,
    rollWindow=3, ylab='EVI')

# produce shape of phenology
outData = '/home/javier/Documents/SF_delta/Sentinel/npphen_consulta/outShape.tif'
    nan_replace=-32767, nGS=46, chuckSize=80, n_jobs=4)

# get land surface phenology metrics
phen.PhenoLSP(inData=outData, outData=outData[:-4]+'_LSP.tif',
    nGS=46, n_jobs=4, chuckSize=256)
