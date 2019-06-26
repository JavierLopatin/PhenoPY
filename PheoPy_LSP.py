
import pandas as pd
import numpy as np
import os
import glob
import phenopy as phen


# list of folders
home = '/home/javier/Documents/SF_delta/Sentinel'
os.chdir(home)
outdir = home+'/LSP'

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


# apply LSP metrics to the list of data
for i in range(len(listTSS)):
    # read dates
    dates = pd.read_csv(listDates[i], header=None)[0]
    dates = pd.to_datetime(dates)

    # get shape of accumulative phenology
    outShape = outdir + '/' + folders[i] + '_phenoshape.tif'
    phen.PhenoShape(inData=listTSS[i], outData=outShape, dates=dates,
        rollWindow=3, nan_replace=-32767, nGS=46, chuckSize=256, n_jobs=4)


for i in range(len(listTSS)):
    # get LSP metrics
    outShape = outdir + '/' + folders[i] + '_phenoshape.tif'
    outLSP = outdir + '/' + folders[i] + '_LSP.tif'
    phen.PhenoLSP(inData=outShape, outData=outLSP, nGS=46, n_jobs=6,
        chuckSize=256)
