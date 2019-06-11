
import pandas as pd
import phenopy as phen
import timeit
import matplotlib.pyplot as plt

days = '/home/javier/Documents/SF_delta/Sentinel/TSA/npphen_consulta/dates.txt'
inData = '/home/javier/Documents/SF_delta/Sentinel/TSA/npphen_consulta/rasterTest.tif'
inData2 = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/2015-2019_001-365_LEVEL4_TSA_SEN2L_EVI_C0_S0_FAVG_TY_C95T_TSS.tif'
outData = '/home/javier/Documents/SF_delta/Sentinel/TSA/npphen_consulta/outTest.tif'
# import dates
dates = pd.read_csv(days, header=None)[0]
dates = pd.to_datetime(dates)
# create dummy set of dates all with same year to mix according to DOY
#years2 = years.apply(lambda dt: dt.replace(year=2018))

# Plot
X = 579470.599
Y = 4224782.793
outFigure = '/home/javier/Documents/SF_delta/Sentinel/TSA/npphen_consulta/testFig.pdf'
phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, saveFigure=outFigure, ylim=None, rollWindow=5, nGS=46, ylab='EVI')

phen.PhenoShape(inData=inData, saveRaster=outData, dates=dates, rollWindow=5, nGS=46, nan_replace=-32767)
