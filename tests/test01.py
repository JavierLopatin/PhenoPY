# For testing inside the Datacube Platform!
%matplotlib inline
%load_ext autoreload
%autoreload 2
import datacube
import numpy as np
import xarray as xr
from datacube.utils.masking import make_mask, mask_invalid_data, describe_variable_flags
from datacube.utils.rio import configure_s3_access
from dask.distributed import Client
from phenoxr.phenoXr import Pheno

# client = Client()
# configure_s3_access(aws_unsigned=False, requester_pays=True, client=client)
configure_s3_access(aws_unsigned=False, requester_pays=True)

dc = datacube.Datacube(app="Pheno_test")
dc.list_products().name

veg_proxy = 'NDVI'
dates = ('2018-01-01', '2020-12-31')
inProduct = 'usgs_aws_ls8c2_sr'
collection = 'c1' # 'c1' (for USGS Collection 1);'c2' (for USGS Collection 2) and 's2' (for Sentinel-2)
resolution = 30
central_lat, central_lon = -35.979288, -72.598012
buffer = 0.005
study_area_lat = (central_lat - buffer, central_lat + buffer)
study_area_lon = (central_lon - buffer, central_lon + buffer)

query = {
    "x": study_area_lon,
    "y": study_area_lat,
    "time": dates,
    "output_crs": "EPSG:32719",
    "resolution": (-resolution, resolution),
    "dask_chunks": {"time": 1},
    "group_by":"solar_day",
    'product': inProduct
}

ds = dc.load(
    **query,
)

# ds.update(ds.assign_coords(doy=ds.time.dt.dayofyear,
#                    year=ds.time.dt.year))

doy=ds.time.dt.dayofyear.values
reflectance_names = ["coastal_aerosol", "blue", "green", "red", "nir", "swir1", "swir2"]

mask_cloud = make_mask(ds.pixel_qa, cloud='not_high_confidence', cloud_shadow='not_high_confidence', nodata=False) # genera máscara de nubes e inválidos (landsat aws)
mask_sat = ds.radsat_qa == 0 # pixeles no saturados
dsf = ds[reflectance_names].where(mask_cloud & mask_sat) 
dsf.update(dsf.where((dsf >= 1) & (dsf <= 65455), np.nan))
dsf.update(dsf * 0.0000275 + -0.2)

ndvi = ((dsf.nir - dsf.red) / (dsf.nir + dsf.red)).persist()

# 3D application
ans = ndvi.pheno.computePheno()

#
ans2 = ans.pheno.computePhenoLSP().persist()


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

