
import pandas as pd
import numpy as np
import os
import glob
import phenopy as phen


# list of folders
home = '/home/javier/Documents/SF_delta/Sentinel'
os.chdir(home+'/TSA_cut')
outdir = home+'/LSP'

"""
folders = [dI for dI in os.listdir('Level2') if os.path.isdir(os.path.join('Level2', dI))]
folders
listTSS = []
listDates = []

for i in range(len(folders)):
    os.chdir(home+'/TSA/'+folders[i])
    pat = glob.glob('*TSS.tif', recursive=False)
    listTSS.append(os.getcwd()+'/'+pat[0])
    listDates.append(os.getcwd()+'/dates.txt')
listTSS
listDates
"""
files = glob.glob('*.tif')
files[0][:-4]

# create phenological shape of the wetland
for i in range(len(files)):

    #dates = pd.read_csv(listDates[i], header=None)[0]
    dates = pd.read_csv(files[i][:-4]+".txt", header=None)[0]
    dates = pd.to_datetime(dates)

    # prepare outputs
    outShape = files[i][:-4] + '_phenoshape.tif'

    # process
    print('Processing PhenoShape of ', files[i])
    phen.PhenoShape(inData=files[i], outData=outShape, dates=dates,
                    rollWindow=5, nan_replace=-32767, nGS=46, chuckSize=128,
                    n_jobs=8)

# apply LSP metrics to the phenoshape created
for i in range(len(files)):

    # read dates
    dates = pd.read_csv(files[i][:-4]+".txt", header=None)[0]
    dates = pd.to_datetime(dates)
    doy = dates.dt.dayofyear

    # prepare data
    outShape = files[i][:-4] + '_phenoshape.tif'
    outLSP = files[i][:-4] + '_LSP.tif'

    # process
    print('Processing LSP of ', outShape)
    phen.PhenoLSP(inData=outShape, outData=outLSP, doy=doy,
                  nGS=46, n_jobs=8, chuckSize=128)

# estimate RMSE between the fitted phenoshape and the real distribution of values
for i in range(len(files)):

    # read dates
    dates = pd.read_csv(files[i][:-4]+".txt", header=None)[0]
    dates = pd.to_datetime(dates)

    # prepare data
    inData = files[i]
    inShape = files[i][:-4] + '_phenoshape.tif'
    outData = files[i][:-4] + '_RMSE.tif'

    # process
    print('Processing LSP of ', outShape)
    phen.RMSE(inData, outData, outData, dates)
