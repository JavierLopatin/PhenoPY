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
- xarrar
- rioxarray
- shapely
- pandas
- numpy
- scipy
- matplotlib
- tqdm


### Documentation comming soon...

### Exampe:

Monthly tome series data of SIF (satellite-retrieved solar-induced  chlorophyll fluorescence) depicted from Chen et al. (2022; https://www.nature.com/articles/s41597-022-01520-1). The area here is a small sample for Chile.


```python
import numpy as np
import sys, os
import xarray as xr
import rioxarray as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# PhenoPy modules
from phenopy import Pheno
from plotting import plot_with_southern_doy, PhenoPlot

```


```python
# load time data corresponding to the time serie data
days = 'data/SIF_sample_dates.csv'
dates = pd.read_csv(days, header=None)[0]
dates = pd.to_datetime(dates)
dates.head()
```




    0   2001-01-01
    1   2001-01-09
    2   2001-01-17
    3   2001-01-25
    4   2001-02-02
    Name: 0, dtype: datetime64[ns]




```python
inData = 'data/SIF_sample.tif'
img = rio.open_rasterio(inData)

# Check if img is an xarray DataArray
if not hasattr(img, 'assign_coords'):
    raise ValueError("img is not an xarray DataArray")

# Check if dates is defined
if 'dates' not in locals():
    raise ValueError("dates variable is not defined")

# Change 'band' dim to 'time' and assign time values to xarray dimension
img = img.rename({'band': 'time'}).assign_coords(time=dates.values)
# Assign day of year and year
img['doy'] = img.time.dt.dayofyear
img['year'] = img.time.dt.year

img
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (time: 920, y: 17, x: 20)&gt;
[312800 values with dtype=float32]
Coordinates:
  * time         (time) datetime64[ns] 2001-01-01 2001-01-09 ... 2020-12-26
  * x            (x) float64 -73.02 -72.97 -72.92 ... -72.17 -72.12 -72.07
  * y            (y) float64 -37.63 -37.68 -37.73 ... -38.33 -38.38 -38.43
    spatial_ref  int64 0
    doy          (time) int64 1 9 17 25 33 41 49 ... 313 321 329 337 345 353 361
    year         (time) int64 2001 2001 2001 2001 2001 ... 2020 2020 2020 2020
Attributes:
    _FillValue:    -3.3999999521443642e+38
    scale_factor:  1.0
    add_offset:    0.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 920</li><li><span class='xr-has-index'>y</span>: 17</li><li><span class='xr-has-index'>x</span>: 20</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-a2a0febc-2521-455c-9e66-02af8f6005ed' class='xr-array-in' type='checkbox' checked><label for='section-a2a0febc-2521-455c-9e66-02af8f6005ed' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>...</span></div><div class='xr-array-data'><pre>[312800 values with dtype=float32]</pre></div></div></li><li class='xr-section-item'><input id='section-2cdc5bfd-d79b-4536-bd85-5e934867e30e' class='xr-section-summary-in' type='checkbox'  checked><label for='section-2cdc5bfd-d79b-4536-bd85-5e934867e30e' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2001-01-01 ... 2020-12-26</div><input id='attrs-20682e24-49ab-4493-ae41-d156e78a0e3f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-20682e24-49ab-4493-ae41-d156e78a0e3f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fd722242-f339-4527-b653-019477715979' class='xr-var-data-in' type='checkbox'><label for='data-fd722242-f339-4527-b653-019477715979' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2001-01-01T00:00:00.000000000&#x27;, &#x27;2001-01-09T00:00:00.000000000&#x27;,
       &#x27;2001-01-17T00:00:00.000000000&#x27;, ..., &#x27;2020-12-10T00:00:00.000000000&#x27;,
       &#x27;2020-12-18T00:00:00.000000000&#x27;, &#x27;2020-12-26T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-73.02 -72.97 ... -72.12 -72.07</div><input id='attrs-88d2b2cd-7357-4372-a391-989731aa8f84' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-88d2b2cd-7357-4372-a391-989731aa8f84' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e17037e7-565e-4c23-997c-dec6e84fc11b' class='xr-var-data-in' type='checkbox'><label for='data-e17037e7-565e-4c23-997c-dec6e84fc11b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-73.025, -72.975, -72.925, -72.875, -72.825, -72.775, -72.725, -72.675,
       -72.625, -72.575, -72.525, -72.475, -72.425, -72.375, -72.325, -72.275,
       -72.225, -72.175, -72.125, -72.075])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-37.63 -37.68 ... -38.38 -38.43</div><input id='attrs-67bfafb0-cc0b-46dd-b541-03fd52887462' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-67bfafb0-cc0b-46dd-b541-03fd52887462' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-80891405-e994-4189-8705-e23cf470eac3' class='xr-var-data-in' type='checkbox'><label for='data-80891405-e994-4189-8705-e23cf470eac3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-37.625, -37.675, -37.725, -37.775, -37.825, -37.875, -37.925, -37.975,
       -38.025, -38.075, -38.125, -38.175, -38.225, -38.275, -38.325, -38.375,
       -38.425])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-5b6c25e8-7b5c-4585-8054-ff333fb541f2' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-5b6c25e8-7b5c-4585-8054-ff333fb541f2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e96f3987-57a2-47cf-8f16-d138f5ad3b51' class='xr-var-data-in' type='checkbox'><label for='data-e96f3987-57a2-47cf-8f16-d138f5ad3b51' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>semi_major_axis :</span></dt><dd>6378137.0</dd><dt><span>semi_minor_axis :</span></dt><dd>6356752.314245179</dd><dt><span>inverse_flattening :</span></dt><dd>298.257223563</dd><dt><span>reference_ellipsoid_name :</span></dt><dd>WGS 84</dd><dt><span>longitude_of_prime_meridian :</span></dt><dd>0.0</dd><dt><span>prime_meridian_name :</span></dt><dd>Greenwich</dd><dt><span>geographic_crs_name :</span></dt><dd>WGS 84</dd><dt><span>grid_mapping_name :</span></dt><dd>latitude_longitude</dd><dt><span>spatial_ref :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>GeoTransform :</span></dt><dd>-73.05 0.04999999999999997 0.0 -37.60000000000001 0.0 -0.05</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>doy</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1 9 17 25 33 ... 337 345 353 361</div><input id='attrs-b4a3021e-2800-4dec-b820-a9787f86ead5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b4a3021e-2800-4dec-b820-a9787f86ead5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0d0241f9-ccb6-4d4d-8cf6-2d2d671466d5' class='xr-var-data-in' type='checkbox'><label for='data-0d0241f9-ccb6-4d4d-8cf6-2d2d671466d5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,  97,
       105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193, 201,
       209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297, 305,
       313, 321, 329, 337, 345, 353, 361,   1,   9,  17,  25,  33,  41,
        49,  57,  65,  73,  81,  89,  97, 105, 113, 121, 129, 137, 145,
       153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249,
       257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345, 353,
       361,   1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,
        97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193,
       201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297,
       305, 313, 321, 329, 337, 345, 353, 361,   1,   9,  17,  25,  33,
        41,  49,  57,  65,  73,  81,  89,  97, 105, 113, 121, 129, 137,
       145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241,
       249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345,
       353, 361,   1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,
        89,  97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185,
       193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289,
       297, 305, 313, 321, 329, 337, 345, 353, 361,   1,   9,  17,  25,
        33,  41,  49,  57,  65,  73,  81,  89,  97, 105, 113, 121, 129,
       137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233,
...
       153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241, 249,
       257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345, 353,
       361,   1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,  89,
        97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185, 193,
       201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289, 297,
       305, 313, 321, 329, 337, 345, 353, 361,   1,   9,  17,  25,  33,
        41,  49,  57,  65,  73,  81,  89,  97, 105, 113, 121, 129, 137,
       145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233, 241,
       249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337, 345,
       353, 361,   1,   9,  17,  25,  33,  41,  49,  57,  65,  73,  81,
        89,  97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177, 185,
       193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281, 289,
       297, 305, 313, 321, 329, 337, 345, 353, 361,   1,   9,  17,  25,
        33,  41,  49,  57,  65,  73,  81,  89,  97, 105, 113, 121, 129,
       137, 145, 153, 161, 169, 177, 185, 193, 201, 209, 217, 225, 233,
       241, 249, 257, 265, 273, 281, 289, 297, 305, 313, 321, 329, 337,
       345, 353, 361,   1,   9,  17,  25,  33,  41,  49,  57,  65,  73,
        81,  89,  97, 105, 113, 121, 129, 137, 145, 153, 161, 169, 177,
       185, 193, 201, 209, 217, 225, 233, 241, 249, 257, 265, 273, 281,
       289, 297, 305, 313, 321, 329, 337, 345, 353, 361])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>year</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>2001 2001 2001 ... 2020 2020 2020</div><input id='attrs-cd6aeb60-b032-4d35-88cd-86d60f176aab' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-cd6aeb60-b032-4d35-88cd-86d60f176aab' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fb8a01b8-5338-41fb-a260-756f9fcd407e' class='xr-var-data-in' type='checkbox'><label for='data-fb8a01b8-5338-41fb-a260-756f9fcd407e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001,
       2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001,
       2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001,
       2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001,
       2001, 2001, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002,
       2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002,
       2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002,
       2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002, 2002,
       2002, 2002, 2002, 2002, 2003, 2003, 2003, 2003, 2003, 2003, 2003,
       2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003,
       2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003,
       2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003, 2003,
       2003, 2003, 2003, 2003, 2003, 2003, 2004, 2004, 2004, 2004, 2004,
       2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004,
       2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004,
       2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004,
       2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2005, 2005, 2005,
       2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005,
       2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005,
       2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005, 2005,
...
       2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016,
       2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016,
       2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2017,
       2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017,
       2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017,
       2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017,
       2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017,
       2017, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018,
       2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018,
       2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018,
       2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018, 2018,
       2018, 2018, 2018, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
       2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
       2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
       2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019,
       2019, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2020, 2020,
       2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020,
       2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020,
       2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020,
       2020, 2020, 2020, 2020, 2020, 2020, 2020])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-54bed8a6-a01d-44f1-b1c6-2fe9599ff20f' class='xr-section-summary-in' type='checkbox'  checked><label for='section-54bed8a6-a01d-44f1-b1c6-2fe9599ff20f' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>_FillValue :</span></dt><dd>-3.3999999521443642e+38</dd><dt><span>scale_factor :</span></dt><dd>1.0</dd><dt><span>add_offset :</span></dt><dd>0.0</dd></dl></div></li></ul></div></div>




```python
img.isel(time=0).plot(robust=True)
```




    <matplotlib.collections.QuadMesh at 0x7ff0e67603d0>




    
![png](ExampleData/output_4_1.png)
    



```python
img.isel(x=5, y=5).plot.line('b-^', figsize=(11,4))
plt.title('SIF timeseries')
plt.ylabel('SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')
```




    Text(0, 0.5, 'SIF [$Wm^{-2}nm^{-1}sr^{-1}$]')




    
![png](ExampleData/output_5_1.png)
    



```python
# plot one year time series
img.where(img.year == 2017, drop=True).isel(x=5, y=5).plot.line('b-^', figsize=(11,4))
plt.ylabel('SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')

```




    Text(0, 0.5, 'SIF [$Wm^{-2}nm^{-1}sr^{-1}$]')




    
![png](ExampleData/output_6_1.png)
    



```python
# plot one year time series by doy
img.where(img.year == 2017, drop=True).isel(x=5, y=5).plot.line('b-^', x='doy', figsize=(11,4))
plt.ylabel('SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')

```




    Text(0, 0.5, 'SIF [$Wm^{-2}nm^{-1}sr^{-1}$]')




    
![png](ExampleData/output_7_1.png)
    


As you can see, if you take only yearly data you may end up with many gaps in the data. So, if you are interested in stable land surface phenology (LSP) metrics, you may want to mix several years together to fill the gaps with other years information:


```python
# Use the PhenoPlot funtion with an example X, Y coordinates to play with interpolation parameters.

X = np.median(img.x.values)
Y = np.median(img.y.values)

PhenoPlot(img, X, Y, interpolType='linear', rollWindow=5, plotType=1)
```




    <AxesSubplot: xlabel='Day of the year', ylabel='NDVI'>




    
![png](ExampleData/output_9_1.png)
    


As we are in the souther hemosphere in this case, the typical order of DOY does not give us the necesary phenological shape for rstimating Land Surface Phenology (LSP) metrics.

Hence, we can create pseudo-DOYs for fitting the necesary data and then correct for the real DOY values when plotting.


```python
### Change the DOY order for southern hemisphere

from utils import reorder_southern_hemisphere

reordered_doy, south_img = reorder_southern_hemisphere(img)
south_img
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (time: 920, y: 17, x: 20)&gt;
[312800 values with dtype=float32]
Coordinates:
  * time         (time) datetime64[ns] 2001-07-04 2002-07-04 ... 2020-06-25
  * x            (x) float64 -73.02 -72.97 -72.92 ... -72.17 -72.12 -72.07
  * y            (y) float64 -37.63 -37.68 -37.73 ... -38.33 -38.38 -38.43
    spatial_ref  int64 0
    doy          (time) int64 1 1 1 1 1 1 1 1 ... 361 361 361 361 361 361 361
    year         (time) int64 2001 2002 2003 2004 2005 ... 2017 2018 2019 2020
Attributes:
    _FillValue:    -3.3999999521443642e+38
    scale_factor:  1.0
    add_offset:    0.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>time</span>: 920</li><li><span class='xr-has-index'>y</span>: 17</li><li><span class='xr-has-index'>x</span>: 20</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-acf8c3d8-20ff-4325-8dd0-1401fb1cb787' class='xr-array-in' type='checkbox' checked><label for='section-acf8c3d8-20ff-4325-8dd0-1401fb1cb787' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>...</span></div><div class='xr-array-data'><pre>[312800 values with dtype=float32]</pre></div></div></li><li class='xr-section-item'><input id='section-f5d29e0c-75e7-45f5-9294-7d8eb40bdeeb' class='xr-section-summary-in' type='checkbox'  checked><label for='section-f5d29e0c-75e7-45f5-9294-7d8eb40bdeeb' class='xr-section-summary' >Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2001-07-04 ... 2020-06-25</div><input id='attrs-157e48af-b16c-4f4e-b557-7a7c4a3859c7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-157e48af-b16c-4f4e-b557-7a7c4a3859c7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-99617d60-1f4d-4d7e-b0c0-562be9a28b96' class='xr-var-data-in' type='checkbox'><label for='data-99617d60-1f4d-4d7e-b0c0-562be9a28b96' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;2001-07-04T00:00:00.000000000&#x27;, &#x27;2002-07-04T00:00:00.000000000&#x27;,
       &#x27;2003-07-04T00:00:00.000000000&#x27;, ..., &#x27;2018-06-26T00:00:00.000000000&#x27;,
       &#x27;2019-06-26T00:00:00.000000000&#x27;, &#x27;2020-06-25T00:00:00.000000000&#x27;],
      dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-73.02 -72.97 ... -72.12 -72.07</div><input id='attrs-d04061bb-9260-4b7f-af7e-ae2cef2dfe71' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d04061bb-9260-4b7f-af7e-ae2cef2dfe71' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-597323ef-025f-4f45-8632-897329793f79' class='xr-var-data-in' type='checkbox'><label for='data-597323ef-025f-4f45-8632-897329793f79' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-73.025, -72.975, -72.925, -72.875, -72.825, -72.775, -72.725, -72.675,
       -72.625, -72.575, -72.525, -72.475, -72.425, -72.375, -72.325, -72.275,
       -72.225, -72.175, -72.125, -72.075])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-37.63 -37.68 ... -38.38 -38.43</div><input id='attrs-84ba9407-4565-49e7-b52e-e2c06f8cd0f4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-84ba9407-4565-49e7-b52e-e2c06f8cd0f4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-58c5bac6-2e7b-4e76-b050-dd2b9e4e39bb' class='xr-var-data-in' type='checkbox'><label for='data-58c5bac6-2e7b-4e76-b050-dd2b9e4e39bb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-37.625, -37.675, -37.725, -37.775, -37.825, -37.875, -37.925, -37.975,
       -38.025, -38.075, -38.125, -38.175, -38.225, -38.275, -38.325, -38.375,
       -38.425])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-08e72242-c782-40bf-b2a3-9064eb4284e7' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-08e72242-c782-40bf-b2a3-9064eb4284e7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4627fc46-8f54-4db7-8c97-6749ccb1dbea' class='xr-var-data-in' type='checkbox'><label for='data-4627fc46-8f54-4db7-8c97-6749ccb1dbea' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>semi_major_axis :</span></dt><dd>6378137.0</dd><dt><span>semi_minor_axis :</span></dt><dd>6356752.314245179</dd><dt><span>inverse_flattening :</span></dt><dd>298.257223563</dd><dt><span>reference_ellipsoid_name :</span></dt><dd>WGS 84</dd><dt><span>longitude_of_prime_meridian :</span></dt><dd>0.0</dd><dt><span>prime_meridian_name :</span></dt><dd>Greenwich</dd><dt><span>geographic_crs_name :</span></dt><dd>WGS 84</dd><dt><span>grid_mapping_name :</span></dt><dd>latitude_longitude</dd><dt><span>spatial_ref :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>GeoTransform :</span></dt><dd>-73.05 0.04999999999999997 0.0 -37.60000000000001 0.0 -0.05</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>doy</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1 1 1 1 1 1 ... 361 361 361 361 361</div><input id='attrs-350fd102-a572-4c8e-a18b-a987631e06c6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-350fd102-a572-4c8e-a18b-a987631e06c6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a2ff24d8-a105-42ce-8d47-1dab5c4ddcfb' class='xr-var-data-in' type='checkbox'><label for='data-a2ff24d8-a105-42ce-8d47-1dab5c4ddcfb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
         1,   1,   1,   1,   1,   1,   1,   9,   9,   9,   9,   9,   9,
         9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,
         9,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,
        17,  17,  17,  17,  17,  17,  17,  17,  25,  25,  25,  25,  25,
        25,  25,  25,  25,  25,  25,  25,  25,  25,  25,  25,  25,  25,
        25,  25,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
        33,  33,  33,  33,  33,  33,  33,  33,  33,  41,  41,  41,  41,
        41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,
        41,  41,  41,  49,  49,  49,  49,  49,  49,  49,  49,  49,  49,
        49,  49,  49,  49,  49,  49,  49,  49,  49,  49,  57,  57,  57,
        57,  57,  57,  57,  57,  57,  57,  57,  57,  57,  57,  57,  57,
        57,  57,  57,  57,  65,  65,  65,  65,  65,  65,  65,  65,  65,
        65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  65,  73,  73,
        73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,
        73,  73,  73,  73,  73,  81,  81,  81,  81,  81,  81,  81,  81,
        81,  81,  81,  81,  81,  81,  81,  81,  81,  81,  81,  81,  89,
        89,  89,  89,  89,  89,  89,  89,  89,  89,  89,  89,  89,  89,
        89,  89,  89,  89,  89,  89,  97,  97,  97,  97,  97,  97,  97,
        97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,
...
       265, 265, 265, 265, 265, 265, 265, 265, 265, 265, 265, 265, 265,
       265, 265, 265, 265, 273, 273, 273, 273, 273, 273, 273, 273, 273,
       273, 273, 273, 273, 273, 273, 273, 273, 273, 273, 273, 281, 281,
       281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
       281, 281, 281, 281, 281, 289, 289, 289, 289, 289, 289, 289, 289,
       289, 289, 289, 289, 289, 289, 289, 289, 289, 289, 289, 289, 297,
       297, 297, 297, 297, 297, 297, 297, 297, 297, 297, 297, 297, 297,
       297, 297, 297, 297, 297, 297, 305, 305, 305, 305, 305, 305, 305,
       305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305,
       313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313,
       313, 313, 313, 313, 313, 313, 313, 321, 321, 321, 321, 321, 321,
       321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321,
       321, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329,
       329, 329, 329, 329, 329, 329, 329, 329, 337, 337, 337, 337, 337,
       337, 337, 337, 337, 337, 337, 337, 337, 337, 337, 337, 337, 337,
       337, 337, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345, 345,
       345, 345, 345, 345, 345, 345, 345, 345, 345, 353, 353, 353, 353,
       353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353, 353,
       353, 353, 353, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361,
       361, 361, 361, 361, 361, 361, 361, 361, 361, 361])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>year</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>2001 2002 2003 ... 2018 2019 2020</div><input id='attrs-3124604a-3411-4eef-9104-e07fb0cd955f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3124604a-3411-4eef-9104-e07fb0cd955f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-20878ae0-29dc-45fa-8ea2-eab8b426534e' class='xr-var-data-in' type='checkbox'><label for='data-20878ae0-29dc-45fa-8ea2-eab8b426534e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
       2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002,
       2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
       2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004,
       2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
       2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006,
       2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
       2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
       2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
       2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
       2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001,
       2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
       2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003,
       2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
       2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005,
       2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
       2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
       2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
       2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
       2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
...
       2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
       2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006,
       2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
       2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
       2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
       2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
       2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001,
       2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
       2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003,
       2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
       2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005,
       2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
       2017, 2018, 2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
       2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
       2019, 2020, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
       2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
       2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
       2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2001, 2002,
       2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
       2014, 2015, 2016, 2017, 2018, 2019, 2020])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-e259d024-a06e-4663-808e-373c537f03b1' class='xr-section-summary-in' type='checkbox'  checked><label for='section-e259d024-a06e-4663-808e-373c537f03b1' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>_FillValue :</span></dt><dd>-3.3999999521443642e+38</dd><dt><span>scale_factor :</span></dt><dd>1.0</dd><dt><span>add_offset :</span></dt><dd>0.0</dd></dl></div></li></ul></div></div>




```python
# plot one year time series by doy in the reordered dataset for southern hemisphere
south_img.where(south_img.year == 2017, drop=True).isel(x=5, y=5).plot.line('b-^', x='doy', figsize=(11,4))
plt.ylabel('SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')
plt.title('SIF timeseries in the southern hemisphere with pseudo-doy')
```




    Text(0.5, 1.0, 'SIF timeseries in the southern hemisphere with pseudo-doy')




    
![png](ExampleData/output_12_1.png)
    


Now we can estimate an interannual phenological shape using all year and heve summer time in the middle of the season.

Using multi-year data, PhenoPy is able to fill the gaps and fit a single phenological shape from which estimate phenological parameters, such as the start of the season (SOS), the peak of season (POS), and the end of season (EOS). You can check if these values are correctly assessed by using type=2 in PhenoPlot:


```python
### TODO: FIX PLOTTING OF DOTS (YEARS) AND INTERPOLATION

# test model parameter using a 2D plot

# example coordinates for testing
X = np.median(img.x.values)
Y = np.median(img.y.values)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the DataArrays side by side
PhenoPlot(south_img, X, Y, 'linear', rollWindow=5,
          plotType=1, ax=axes[0], ylab='SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')

PhenoPlot(south_img, X, Y, 'linear', rollWindow=5,
          plotType=2, ax=axes[1], ylab='SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')

axes[0].set_title("Plot all observations")
axes[1].set_title("Plot SOS, POS, and EOS estimations")

# Display the plots
plt.show()

```


    
![png](ExampleData/output_14_0.png)
    


Estimate Phenological shape to the raster using the same parameters. Here we are using weekly interpolations (nGS=52), linear interpolation and a rollWindow of 5 for smoothing.


```python
# PhenoShape
shape = img.pheno.PhenoShape(rollWindow=5)
shape_south = south_img.pheno.PhenoShape(rollWindow=5)
```


```python
# plot with selected xy coordinates and use the closest real values if the selected xy is not in the original data
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the DataArrays side by side
shape.sel(x=X, y=Y, method="nearest").plot(ax=axes[0])
shape_south.sel(x=X, y=Y, method="nearest").plot(ax=axes[1])

axes[0].set_title("Without changing hemisphere")
axes[1].set_title("Changing hemisphere")

# Display the plots
plt.show()
```


    
![png](ExampleData/output_17_0.png)
    



```python
# reorder xlabels for southern hemisphere
plot_with_southern_doy(shape_south, coordinates=[X,Y], ylabel='SIF ['r'$Wm^{-2}nm^{-1}sr^{-1}$]')
```




    <AxesSubplot: title={'center': 'y = -38.03, x = -72.52'}, xlabel='doy', ylabel='SIF [$Wm^{-2}nm^{-1}sr^{-1}$]'>




    
![png](ExampleData/output_18_1.png)
    


We can estimate the LSP metrics for the raster. Until now, the metrics included are:

SOS - DOY of Start of season \
POS - DOY of Peak of season \
EOS - DOY of End of season \
vSOS - Vaues at start os season \
vPOS - Values at peak of season \
vEOS - Values at end of season \
LOS - Length of season \
MSP - Mid spring (DOY) \
MAU - Mid autum (DOY) \
vMSP - Value at mean spring \
vMAU - Value at mean autum \
AOS - Amplitude of season \
IOS - Integral of season [SOS-EOS] \
ROG - Rate of greening [slope SOS-POS] \
ROS - Rate of senescence [slope POS-EOS] \
SW - Skewness of growing season [SOS-EOS]


```python
# Land surface phenology metrics (LSP)
lsp = shape_south.pheno.PhenoLSP()
lsp
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (y: 17, x: 20)
Coordinates:
  * y        (y) float64 -37.63 -37.68 -37.73 -37.78 ... -38.33 -38.38 -38.43
  * x        (x) float64 -73.02 -72.97 -72.92 -72.87 ... -72.17 -72.12 -72.07
Data variables: (12/16)
    sos      (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    pos      (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    eos      (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    vsos     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    vpos     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    veos     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    ...       ...
    vmau     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    ampl     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    ios      (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    rog      (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    ros      (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    sw       (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-dee2c44c-2a1b-4702-a6cc-9d80edf18f01' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-dee2c44c-2a1b-4702-a6cc-9d80edf18f01' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>y</span>: 17</li><li><span class='xr-has-index'>x</span>: 20</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-8757a229-0285-4019-9cbe-9c19682fd026' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8757a229-0285-4019-9cbe-9c19682fd026' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-37.63 -37.68 ... -38.38 -38.43</div><input id='attrs-6409511c-1493-4dae-a17e-53fe885772e5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6409511c-1493-4dae-a17e-53fe885772e5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0fb43781-577c-42c9-be33-53f8a442864d' class='xr-var-data-in' type='checkbox'><label for='data-0fb43781-577c-42c9-be33-53f8a442864d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-37.625, -37.675, -37.725, -37.775, -37.825, -37.875, -37.925, -37.975,
       -38.025, -38.075, -38.125, -38.175, -38.225, -38.275, -38.325, -38.375,
       -38.425])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-73.02 -72.97 ... -72.12 -72.07</div><input id='attrs-eb021c05-5d10-420e-82b0-1041c20ea7c1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eb021c05-5d10-420e-82b0-1041c20ea7c1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-86c9971f-a54a-49a4-8635-de0effb2f1cc' class='xr-var-data-in' type='checkbox'><label for='data-86c9971f-a54a-49a4-8635-de0effb2f1cc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-73.025, -72.975, -72.925, -72.875, -72.825, -72.775, -72.725, -72.675,
       -72.625, -72.575, -72.525, -72.475, -72.425, -72.375, -72.325, -72.275,
       -72.225, -72.175, -72.125, -72.075])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-6993d624-342f-4203-9878-3b9f8b883660' class='xr-section-summary-in' type='checkbox'  ><label for='section-6993d624-342f-4203-9878-3b9f8b883660' class='xr-section-summary' >Data variables: <span>(16)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>sos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-f8581015-1d74-4db2-b6ba-37fdb86ef96f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f8581015-1d74-4db2-b6ba-37fdb86ef96f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c60d8c4e-60da-481a-a2e4-830e27d56bd3' class='xr-var-data-in' type='checkbox'><label for='data-c60d8c4e-60da-481a-a2e4-830e27d56bd3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>pos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-b8389a36-909c-4df1-824b-6a55a8fe8676' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b8389a36-909c-4df1-824b-6a55a8fe8676' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-645f789f-afbe-419f-9209-75719ea9c035' class='xr-var-data-in' type='checkbox'><label for='data-645f789f-afbe-419f-9209-75719ea9c035' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>eos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-076e14bd-95c8-49e0-a6be-491ff08511a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-076e14bd-95c8-49e0-a6be-491ff08511a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9891af91-3841-4033-93f8-97f11b825cd1' class='xr-var-data-in' type='checkbox'><label for='data-9891af91-3841-4033-93f8-97f11b825cd1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>vsos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-180608f1-796a-4f8c-8e3a-36e35609dc33' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-180608f1-796a-4f8c-8e3a-36e35609dc33' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6a3668f9-7621-4a34-8804-02bb290bbacf' class='xr-var-data-in' type='checkbox'><label for='data-6a3668f9-7621-4a34-8804-02bb290bbacf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>vpos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-286d7999-beb9-46b4-aa3f-7b6cf6f9af36' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-286d7999-beb9-46b4-aa3f-7b6cf6f9af36' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-39db18bb-946f-41ca-b114-ba3185b1548b' class='xr-var-data-in' type='checkbox'><label for='data-39db18bb-946f-41ca-b114-ba3185b1548b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>veos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-d30eae4e-8163-4928-9a5c-3f0381438aa8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d30eae4e-8163-4928-9a5c-3f0381438aa8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9ad36130-c524-4e0b-a6a1-6603e88d5a2e' class='xr-var-data-in' type='checkbox'><label for='data-9ad36130-c524-4e0b-a6a1-6603e88d5a2e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>los</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-c11681a8-c294-41d1-9438-6f97e8eb07d1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c11681a8-c294-41d1-9438-6f97e8eb07d1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-61f361fe-e50d-4ce8-b150-cf8d6cc5ec83' class='xr-var-data-in' type='checkbox'><label for='data-61f361fe-e50d-4ce8-b150-cf8d6cc5ec83' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>msp</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-fa14312d-ddef-46e1-b364-cf11c5f7dc9a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fa14312d-ddef-46e1-b364-cf11c5f7dc9a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e2099fce-8302-45d6-83bd-686894e6316d' class='xr-var-data-in' type='checkbox'><label for='data-e2099fce-8302-45d6-83bd-686894e6316d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>mau</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-ee23a403-575b-491d-99ab-c1f78958e9a6' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ee23a403-575b-491d-99ab-c1f78958e9a6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-11fbb790-32c6-451c-96b6-3e4bd298fcdc' class='xr-var-data-in' type='checkbox'><label for='data-11fbb790-32c6-451c-96b6-3e4bd298fcdc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>vmsp</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-a1ce097b-2837-4c5a-8a97-116d42df7c40' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a1ce097b-2837-4c5a-8a97-116d42df7c40' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9d034d8d-9ad6-45d1-b878-528cc89eec04' class='xr-var-data-in' type='checkbox'><label for='data-9d034d8d-9ad6-45d1-b878-528cc89eec04' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>vmau</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-5e7b407e-626c-40d9-ab9c-afb8d487a907' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5e7b407e-626c-40d9-ab9c-afb8d487a907' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2bdb4289-2128-4570-bdf8-4d32f5780f96' class='xr-var-data-in' type='checkbox'><label for='data-2bdb4289-2128-4570-bdf8-4d32f5780f96' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ampl</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-c968876b-24dc-486b-a326-5e01cf54c7e1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c968876b-24dc-486b-a326-5e01cf54c7e1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-511cdbd1-4067-4d73-9326-aeec9c81c94d' class='xr-var-data-in' type='checkbox'><label for='data-511cdbd1-4067-4d73-9326-aeec9c81c94d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ios</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-5265cbd5-dc53-48b3-8543-af33fdb56715' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5265cbd5-dc53-48b3-8543-af33fdb56715' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-280ff538-844b-401f-a459-86bf756a380b' class='xr-var-data-in' type='checkbox'><label for='data-280ff538-844b-401f-a459-86bf756a380b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rog</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-0a365b25-b833-4a09-b468-792565fdf436' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0a365b25-b833-4a09-b468-792565fdf436' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ddf6aeee-da30-4286-baae-c3000883df3e' class='xr-var-data-in' type='checkbox'><label for='data-ddf6aeee-da30-4286-baae-c3000883df3e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>ros</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-60a2fffd-3af2-48a4-b164-2bb3131e687d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-60a2fffd-3af2-48a4-b164-2bb3131e687d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-878657ef-4ccc-4f4b-9728-211133f5ab5b' class='xr-var-data-in' type='checkbox'><label for='data-878657ef-4ccc-4f4b-9728-211133f5ab5b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sw</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-a7e0164c-2b82-4b30-ba2a-9f2e81c77e21' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a7e0164c-2b82-4b30-ba2a-9f2e81c77e21' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0f39ee44-8cc8-40e2-9c99-fd6079d0f2f6' class='xr-var-data-in' type='checkbox'><label for='data-0f39ee44-8cc8-40e2-9c99-fd6079d0f2f6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 11 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li></ul></div></li><li class='xr-section-item'><input id='section-78d0655e-f1e0-41b0-b443-5eb412818ee7' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-78d0655e-f1e0-41b0-b443-5eb412818ee7' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
# plot lsp.sos.plot(robust=True) and lsp.eos.plot(robust=True) in the same figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the DataArrays side by side
lsp.sos.plot(ax=axes[0], robust=True)
lsp.eos.plot(ax=axes[1], robust=True)

axes[0].set_title("SOS")
axes[1].set_title("EOS")

# Display the plots
plt.show()    

```


    
![png](ExampleData/output_21_0.png)
    


We can also estimate the interannual variability of the phenological metrics by calculating the root mean square error of the sognal.

See Lopatin (2023; https://ieeexplore.ieee.org/document/10128132).

E.g., 

<img src="phenoshape/data/Fig_Segmented_nRMSE.jpg" alt="" width="700"/>


```python
rmse = shape.pheno.RMSE(img, LSP_stack=lsp, normalized=True)
rmse2 = shape.pheno.RMSE(img, LSP_stack=lsp, normalized=False)
rmse
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:      (x: 20, y: 17)
Coordinates:
  * x            (x) float64 -73.02 -72.97 -72.92 ... -72.17 -72.12 -72.07
  * y            (y) float64 -37.63 -37.68 -37.73 ... -38.33 -38.38 -38.43
    spatial_ref  int64 0
Data variables:
    rmse         (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    rmse_sos     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    rmse_pos     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;
    rmse_eos     (y, x) float64 dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-95f36e55-2d53-4dd6-8830-199e97ba5545' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-95f36e55-2d53-4dd6-8830-199e97ba5545' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 20</li><li><span class='xr-has-index'>y</span>: 17</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-54f46748-4b73-4962-9b75-5492f566fb48' class='xr-section-summary-in' type='checkbox'  checked><label for='section-54f46748-4b73-4962-9b75-5492f566fb48' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-73.02 -72.97 ... -72.12 -72.07</div><input id='attrs-0ed9b747-31a6-4757-88ae-836a8bf4075c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0ed9b747-31a6-4757-88ae-836a8bf4075c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e9bf6ff8-c108-484a-bda7-5a4861949dd9' class='xr-var-data-in' type='checkbox'><label for='data-e9bf6ff8-c108-484a-bda7-5a4861949dd9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-73.025, -72.975, -72.925, -72.875, -72.825, -72.775, -72.725, -72.675,
       -72.625, -72.575, -72.525, -72.475, -72.425, -72.375, -72.325, -72.275,
       -72.225, -72.175, -72.125, -72.075])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-37.63 -37.68 ... -38.38 -38.43</div><input id='attrs-7f1fe980-a421-41db-bbdf-41692e90e82f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7f1fe980-a421-41db-bbdf-41692e90e82f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b3742573-07f9-454a-9927-49896480766e' class='xr-var-data-in' type='checkbox'><label for='data-b3742573-07f9-454a-9927-49896480766e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-37.625, -37.675, -37.725, -37.775, -37.825, -37.875, -37.925, -37.975,
       -38.025, -38.075, -38.125, -38.175, -38.225, -38.275, -38.325, -38.375,
       -38.425])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-d11e2243-3b2a-430a-91a6-122ab75eab8e' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-d11e2243-3b2a-430a-91a6-122ab75eab8e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4c1dd3e3-8bef-4feb-ada3-59125f931b0b' class='xr-var-data-in' type='checkbox'><label for='data-4c1dd3e3-8bef-4feb-ada3-59125f931b0b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>semi_major_axis :</span></dt><dd>6378137.0</dd><dt><span>semi_minor_axis :</span></dt><dd>6356752.314245179</dd><dt><span>inverse_flattening :</span></dt><dd>298.257223563</dd><dt><span>reference_ellipsoid_name :</span></dt><dd>WGS 84</dd><dt><span>longitude_of_prime_meridian :</span></dt><dd>0.0</dd><dt><span>prime_meridian_name :</span></dt><dd>Greenwich</dd><dt><span>geographic_crs_name :</span></dt><dd>WGS 84</dd><dt><span>grid_mapping_name :</span></dt><dd>latitude_longitude</dd><dt><span>spatial_ref :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>GeoTransform :</span></dt><dd>-73.05 0.04999999999999997 0.0 -37.60000000000001 0.0 -0.05</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a91f4bf1-f9fc-48d5-ad29-43854f787ed7' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a91f4bf1-f9fc-48d5-ad29-43854f787ed7' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>rmse</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-940cb696-7030-43d2-a174-770b9d52e7ae' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-940cb696-7030-43d2-a174-770b9d52e7ae' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-046728a2-7a8c-46a3-92bd-16c429a64955' class='xr-var-data-in' type='checkbox'><label for='data-046728a2-7a8c-46a3-92bd-16c429a64955' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 28 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rmse_sos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-697b8459-bd18-4c15-a412-d1abb006d324' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-697b8459-bd18-4c15-a412-d1abb006d324' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a42a4f33-30f2-44ee-868b-f4c905cddad6' class='xr-var-data-in' type='checkbox'><label for='data-a42a4f33-30f2-44ee-868b-f4c905cddad6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 43 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rmse_pos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-24f484d7-e70c-439b-a575-9bc041db5ab3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-24f484d7-e70c-439b-a575-9bc041db5ab3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-396bccb5-bbfe-44e3-8a14-3144c8034257' class='xr-var-data-in' type='checkbox'><label for='data-396bccb5-bbfe-44e3-8a14-3144c8034257' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 47 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>rmse_eos</span></div><div class='xr-var-dims'>(y, x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(10, 10), meta=np.ndarray&gt;</div><input id='attrs-664e3cf8-95ae-402c-929b-ba726551da14' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-664e3cf8-95ae-402c-929b-ba726551da14' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f4970fef-cf6d-4b50-811b-fbb50bd40c0c' class='xr-var-data-in' type='checkbox'><label for='data-f4970fef-cf6d-4b50-811b-fbb50bd40c0c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.66 kiB </td>
                        <td> 800 B </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (17, 20) </td>
                        <td> (10, 10) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 43 Graph Layers </td>
                        <td> 4 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float64 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="170" height="152" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="120" y2="0" style="stroke-width:2" />
  <line x1="0" y1="60" x2="120" y2="60" />
  <line x1="0" y1="102" x2="120" y2="102" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="102" style="stroke-width:2" />
  <line x1="60" y1="0" x2="60" y2="102" />
  <line x1="120" y1="0" x2="120" y2="102" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 120.0,0.0 120.0,102.0 0.0,102.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="60.000000" y="122.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >20</text>
  <text x="140.000000" y="51.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,140.000000,51.000000)">17</text>
</svg>
        </td>
    </tr>
</table></div></li></ul></div></li><li class='xr-section-item'><input id='section-909569b6-20c1-47eb-b014-9e9bd4b2619d' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-909569b6-20c1-47eb-b014-9e9bd4b2619d' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>




```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the DataArrays side by side
rmse.rmse.plot(ax=axes[0], robust=True)
rmse2.rmse.plot(ax=axes[1], robust=True)

axes[0].set_title("Overall normalized RMSE")
axes[1].set_title("Overall RMSE")

# Display the plots
plt.show()   
```


    
![png](ExampleData/output_24_0.png)
    



```python
# plot lsp.sos.plot(robust=True) and lsp.eos.plot(robust=True) in the same figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))

# Plot the DataArrays side by side
rmse.rmse_sos.plot(ax=axes[0], robust=True)
rmse.rmse_pos.plot(ax=axes[1], robust=True)
rmse.rmse_eos.plot(ax=axes[2], robust=True)

axes[0].set_title("RMSE_SOS")
axes[1].set_title("RMSE_POS")
axes[2].set_title("RMSE_EOS")

# Display the plots
plt.show()  
```


    
![png](ExampleData/output_25_0.png)
    



```python
# plot RMSE RGB-based figure
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot the DataArrays side by side
rmse.to_array()[[3, 2, 1], :, :].plot.imshow(ax=axes[0], robust=True)
rmse2.to_array()[[3, 2, 1], :, :].plot.imshow(ax=axes[1], robust=True)

axes[0].set_title("Overall normalized RMSE")
axes[1].set_title("Overall RMSE")

# Display the plots
plt.show()
```


    
![png](ExampleData/output_26_0.png)
    

