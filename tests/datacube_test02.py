# For testing inside the Datacube Platform!
%matplotlib inline
%load_ext autoreload
%autoreload 2
import datacube
import numpy as np
import sys
import xarray as xr
import matplotlib.pyplot as plt
from datacube.utils.masking import make_mask, mask_invalid_data, describe_variable_flags
from datacube.utils.rio import configure_s3_access
from datacube.utils.cog import write_cog
from dask.distributed import Client
import geopandas as gpd
import rioxarray as rio

sys.path.append('../')
from phenoxr.phenoXr import Pheno

# client = Client()
# configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
configure_s3_access(aws_unsigned=False, requester_pays=True)

dc = datacube.Datacube(app="Pheno_test")
dc.list_products().name

lat_long = gpd.read_file("shp/P.shp")
lat_long = lat_long.to_crs("EPSG:4326")

xmin, ymin, xmax, ymax = lat_long.total_bounds
time_extents = ('2019-01-01', '2020-12-31')
resolution = 30
inProduct = 'landsat8_c2l2_sr'

query = {
    "x": [xmin, xmax],
    "y": [ymin, ymax],
     "time": time_extents,
    "output_crs": "EPSG:32719",
    "resolution": (-resolution, resolution),
    "dask_chunks": {"time": 1},
    "group_by":"solar_day",
    'product': inProduct,
    'skip_broken_datasets': True
}

ds = dc.load(
    **query,
)

# ds.update(ds.assign_coords(doy=ds.time.dt.dayofyear,
                    # year=ds.time.dt.year))

doy=ds.time.dt.dayofyear.values
reflectance_names = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]

mask_cloud = make_mask(ds['qa_pixel'], cloud='not_high_confidence', cloud_shadow='not_high_confidence', nodata=False) # genera máscara de nubes e inválidos (landsat aws)
mask_sat = ds['qa_radsat'] == 0 # pixeles no saturados
dsf = ds[reflectance_names].where(mask_cloud & mask_sat) 
dsf.update(dsf.where((dsf >= 1) & (dsf <= 65455), np.nan))
dsf.update(dsf * 0.0000275 + -0.2)

ndvi = ((dsf.nir08 - dsf.red) / (dsf.nir08 + dsf.red)).persist()
ndvi
# PhenoShape
ans = ndvi.pheno.PhenoShape(chunk_size={'x': 400, 'y': 400, 'time': len(ndvi.time)})
ans

# save to disk
write_cog(ndvi, 'out/phenopetorca.tif', overwrite=True).compute()

# PhenoShape
ans_rmse = ans.pheno.RMSE(ndvi).persist()
ans_rmse 
ans_rmse.plot(robust=True, figsize=(10, 8))


# Pheno LSP
ans2 = ans.pheno.PhenoLSP().persist()
ans_rmse_c = ans.pheno.RMSE(ndvi, LSP_stack=ans2)




if False:
    ## ------- OLD testing, do not run if not necessary ------
    # Get 1 pixel trought time
    
    onePix = ndvi.sel(x=ndvi.x[1].values, y=ndvi.x[1].values, method="nearest").values

    # 1D application vs time
    onePixr = _getPheno0(y=onePix, doy=doy, interpolType='linear', nan_replace = None, rollWindow=5, nGS = 52)

    # 2D application vs time
    data = _getPheno2(dstack=ndvi.values, doy=doy, 
                      interpolType='linear', nan_replace = None, rollWindow=5, nGS=52)
    ## --------------------------------------------------

    da = xr.DataArray(
        np.sin(0.3 * np.arange(12).reshape(4, 3)),
        [("time", np.arange(4)), ("space", [0.1, 0.2, 0.3])],
        )


    da.sel(time=3)

    da.interp(time=2.5)