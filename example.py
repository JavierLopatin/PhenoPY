import os
import pandas as pd
import phenopy as phen
import timeit
# in data
os.chdir('/home/javier/Documents/SF_delta/Sentinel/TSA/npphen_consulta')
days = 'dates.txt'
inData = 'rasterTest.tif'
inData2 = '/home/javier/Documents/SF_delta/Sentinel/TSA/X-004_Y-001/2015-2019_001-365_LEVEL4_TSA_SEN2L_EVI_C0_S0_FAVG_TY_C95T_TSS.tif'
# import dates
dates = pd.read_csv(days, header=None)[0]
dates = pd.to_datetime(dates)
# create dummy set of dates all with same year to mix according to DOY
#years2 = years.apply(lambda dt: dt.replace(year=2018))

start = timeit.timeit()
phen.PhenoShape(inData, 'test.tif', dates=dates, rollWindow=3, nGS=46, nan_replace=-32767)
end  = timeit.timeit()
print(end - start)
