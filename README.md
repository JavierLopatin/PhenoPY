# PhenoPY
<h1 align="center">
<a href='https://github.com/JavierLopatin/PhenoPY'><img src='data/logo.svg' align="right" height="300" /></a>

<h4 align="center">Python bindings for Phenological analysis of Remote Sensing data </h4>

<p align="center">
  <a href="http://forthebadge.com">
    <img src="http://forthebadge.com/images/badges/made-with-python.svg"
         alt="Gitter">
  </a>
  <a href="http://forthebadge.com"><img src="http://forthebadge.com/images/badges/built-with-love.svg"></a>
  <a href="http://forthebadge.com">
      <img src="http://forthebadge.com/images/badges/built-with-science.svg">
  </a>
</p>

### Library dependencies:
- Python < 3.6
- rasterstats
- rasterio
- shapely
- pandas
- numpy
- scipy
- matplotlib
- tqdm


### Documentation comming soon...

### Exampe:

This example shows the phenology of wetland vegetation in the San Francisco delta, California, USA, using three consecutive years of Sentinel-2 data. With a total of ~120 observations per pixel, a tipical temporal figure looks like:

![alt text](data/timeseries_allData.svg)

As you can see, if you take only yearly data you may end up with many gaps in the data. So, if you are interested in stable land surface phenology (LSP) metrics, you may want to mix several years together to fill the gaps with other years information:

``` python
import phenopy as phen

# coordinates of the pixel to check
X = 584543.89281
Y = 4228482.52520

# plot PhenoShape with multi-year data
phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, ylim=None, type=1,
    rollWindow=3, ylab='EVI')

```
![alt text](data/figure.svg)

Using multi-year data, `PhenoPy` is able to fill the gaps and fit a single phenological shape from which estimate phenological parameters, such as the start of the season (SOS), the peak of season (POS), and the end of season (EOS). You can check if these values are correctly assessed by using `type=2` in `PhenoPlot`:


``` python

# plot PhenoShape with SOS, POS, and EOS locations
phen.PhenoPlot(X=X, Y=Y, inData=inData, dates=dates, ylim=None, type=1,
    rollWindow=3, ylab='EVI')

```

![alt text](data/figure2.svg)

This is only for visualization. A raster of fitted lines per pixel can be achieve:

``` python

phen.PhenoShape(inData=inData, outData=outData, doy=doy, rollWindow=5,
                nan_replace=-32767, nGS=46, chuckSize=256, n_jobs=4)

```
    100%|██████████| 42/42 [00:03<00:00, 13.59it/s]


where `nan_replace` is the NaN value (default None), `nGS` is the munber of interpolation values to use (here we obtained one avarage value per week, hence = 46), `chuckSize` is the size of the chunck that rasterio will open at the time, and `n_jobs` is the number of parallel process to use. We used rasterio to open big images in small-sized chuncks and therefore keep the memory usage low. See example tile below:


![alt text](data/example_phenoshape.png)


Meanwhile, a raster of Land Surface Phenology (LSP) metrics can be obtained as:

``` python
phen.PhenoLSP(inData=outData, outData=outData[:-4] + '_LSP2.tif', doy=doy, phentype=2,
              nGS=46, n_jobs=4, chuckSize=256)
```
   100%|██████████| 8924/8924 [00:01<00:00, 7048.13it/s]
   
Until now, the metrics included are:

- SOS - DOY of Start of season
- POS - DOY of Peak of season
- EOS - DOY of End of season
- vSOS - Vaues at start os season
- vPOS - Values at peak of season
- vEOS - Values at end of season
- LOS - Length of season
- MSP - Mid spring (DOY)
- MAU - Mid autum (DOY)
- vMSP - Value at mean spring
- vMAU - Value at mean autum
- AOS - Amplitude of season
- IOS - Integral of season [SOS-EOS]
- ROG - Rate of greening [slope SOS-POS]
- ROS - Rate of senescence [slope POS-EOS]
- SW - Skewness of growing season [SOS-EOS]

Finally, the library has a estimation of the interannual variation of data, using normalized RMSE values:

```python
# get RMSE between the fitted phenoshape and the real distribution of values
phen.RMSE(inData, outData, outData[:-4] + '_RMSE.tif', dates)
```

    100%|██████████| 8924/8924 [00:01<00:00, 7048.13it/s]

![png](data/output_14_1.png)

TODO:

Add implementation of a segmented nRMSE estimations, where error estimates (i.e., interanual variation) is estimated per segment. Example:


![png](data/Fig_Segmented_nRMSE.jpg)
