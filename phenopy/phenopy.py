###############################################################################
#
# PhenoPy is a Python 3.X library to process phenology indices derived from
# EarthObservation data.
#
###############################################################################

# libraries included in Python 3.X
from __future__ import division
import concurrent.futures
from functools import partial
import sys
import warnings

# common dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error

# speciel dependencies
import xarray as xr                  # manipulate 3D time-series rasters
import shapely.geometry as geom      # create geographical points
import rasterio                      # manipulate GeoTIFF
from rasterstats import point_query  # extract raster values
from tqdm import tqdm                # progress bar
# from kneed import KneeLocator        # find inflection point on a curve
# from KDEpy import FFTKDE             # perform fast 2D kernel density estimations

# from PhenoPy

#import all function from utils.py
from utils import _getPheno0, _getPheno2D, _parseLSP, _getLSPmetrics2, _rmse, _replaceElements
from curvature import get_curvature


@xr.register_dataarray_accessor("pheno")
class Pheno:
    def __init__(self, xr_obj):
        self._obj = xr_obj
        self.kwargs = {}
        self.LSP_bands = ['sos', 'pos', 'eos', 'vsos', 'vpos', 
                          'veos', 'los', 'msp', 'mau', 'vmsp', 
                          'vmau', 'ampl', 'ios', 'rog', 'ros', 'sw']
       
    def PhenoShape(self, interpolType='linear', nan_replace=None,
                   rollWindow=5, nGS=52):
        """
        Apply the _getPheno2D/_getPheno2 function to a xarray.DataArray object. It calculates all the necessary auxiliary objects in order to use Dask functionality (trough map_blocks).
        
        :param doy: day of the year. If not present, it will attempt to extract the doy from the time dimension.
        :param interpolType: interpolation type
        :param nan_replace: what to do with NaNs
        :param rollWindow: rolling window size
        :param nGS: number of periods per year, usually 52 (number of weeks in a year)
        
        :returns: computed xarray.DataArray
        """
        stack = self._obj
        doy=stack.doy.values
        dates=stack.time
  
        # Get the indices that would sort the doy coordinate
        #sorted_indices = np.argsort(stack.doy.values)
        # Reorder the time dimension using the sorted indices
        #stack = stack.isel(time=sorted_indices)
        # replicating what happens in _getPheno xnew definition
        xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype=np.int32)
        # TODO: change hemisfere, start doy at the desired day (1 north, 182 south) and keep record about the original doy -> dos (day of season)
        
        # TODO: define a function to auto calculate next chunk (to ~100MB each chunk)
        time_chunk = {'x': 10, 'y': 10, 'time': len(stack.time)}
        
        coords_ = {'time': xnew,
                  'y': stack.coords['y'],
                  'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((nGS, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['time', 'y', 'x']).chunk(time_chunk)
        
        stack = stack.chunk(time_chunk)
        kwargs_ = {'doy': doy, 'interpolType': interpolType,  
                   'nan_replace': nan_replace, 'rollWindow': rollWindow, 
                   'nGS': nGS, 'xnew': xnew}
        
        stackP = stack.map_blocks(_getPheno2D, kwargs=kwargs_, template=template_).rename({'time': 'doy'})
        stackP.pheno.kwargs['computePheno'] = kwargs_
        
        return stackP
    
    def PhenoLSP(self, nGS=None, phentype=1):
        """
        Obtain land surface phenology metrics for a PhenoShape product

        Parameters
        ----------
        - inData: String
            Absolute path to PhenoShape raster data
        - outData: String
            Absolute path for output land surface phenology raster
        - doy: 1D array
            Numpy array of the days of the year of the time series
        - nGS: Integer
            Number of observations to predict the PhenoShape
            default is 46; one per week
        - phenType: Type os estimation of SOS and EOS. 1 = median value between POS and start and end of season. 2 = using the knee inflexion method. default 1
       
        """
        if 'computePheno' not in self.kwargs:
            raise('No Pheno computed')  # TODO: replace with auto-compute?
        
        n_ = len(self.LSP_bands)
        stack = self._obj
        xnew = self.kwargs['computePheno']['xnew']
        time_chunk = [i for i in stack.chunks]
        time_chunk[0] = n_
        if nGS is None:
            nGS = self.kwargs['computePheno']['nGS']
        
        kwargs_ = {'xnew': xnew, 'nGS': nGS, 'bands': self.LSP_bands, 'phentype': phentype}
        
        coords_ = {'doy': self.LSP_bands,
                   'y': stack.coords['y'],
                   'x': stack.coords['x']}
        template_ = xr.DataArray(np.zeros((n_, len(stack.y), len(stack.x))), 
                                 coords=coords_,
                                 dims = ['doy', 'y', 'x']).chunk(time_chunk)
        
        stackP = stack.map_blocks(_parseLSP, kwargs=kwargs_, template=template_).rename({'doy': 'LSP_bands'})
        stackP.pheno.kwargs['computePhenoLSP'] = kwargs_  # this doesn't work, is not saved, pheno objet its lost in datadaset transformation

        return stackP.to_dataset('LSP_bands')
    
    def RMSE(self, original_stack, LSP_stack, normalized=False, nan_replace=None, interpolate_nans=False, segment=False):
        """
        Calculate the RMSE of the PhenoShape estimation; it can also do it by section, using the sos, pos and eos from the LSP computation (if provided).
        
        :param original_stack: initial image stack, from which the phenoShape was calculated.
        :param LSP_stack: LSP computed stack.
        :param normalized: boolean, if True RMSE will be scaled to [0, 1].
        :param nan_replace: values to be converted to NaN.
        :param interpolate_nans: boolean, should NaNs values be interpolated?
        :param metrics: string, either 'overall' or 'segmented'. If 'overall', the RMSE is calculated for the whole stack. If 'segmented', the RMSE is calculated for the sos, pos and eos segments.
        :returns: computed xarray.DataArray with the RMSE
        """
        # shape = ans.copy(); original_stack=ndvi.copy(); LSP_stack = ans2.copy()
        computed_stack = self._obj # inShape, phen  || # original_stack = inData = dstack
        
        # 1. Check if I'm PhenoShape data
        if 'computePheno' not in self.kwargs:
            raise('It seems computePheno has not yet been computed...')
        
        if nan_replace is not None:
            original_stack = original_stack.where(original_stack.values != nan_replace)
        
        # 2. Get day of the year of original_stack and reorder by that
        doys = original_stack.time.dt.dayofyear.values
        sdoys = sorted(doys)
        original_stack = original_stack.assign_coords(time=doys)
        if 'doy' in original_stack.coords or 'doy' in original_stack.dims:
            pass
        else:
            original_stack = original_stack.rename({'time': 'doy'})

        original_stack = original_stack.sortby('doy')
                  
        # 3. Linear interpolation for pheno, to match original_stack doys
        if interpolate_nans:
            computed_stack = computed_stack.interpolate_na('doy')
            
        computed_stack = computed_stack.interp(doy=sdoys, method='linear')
        
        # If the dimension 'time' exists and 'doy' doesn't, rename 'time' to 'doy'
        if 'time' in original_stack.dims and 'doy' not in original_stack.dims:
            if 'doy' in original_stack.coords:
                original_stack = original_stack.drop_vars('doy')
            original_stack = original_stack.rename({'time': 'doy'})

        # if 'time' in computed_stack.dims and 'doy' not in computed_stack.dims:
        #     if 'doy' in computed_stack.coords:
        #         computed_stack = computed_stack.drop_vars('doy')
        #     computed_stack = computed_stack.rename({'time': 'doy'})

        # Ensure the dimension names are now consistent between original_stack and computed_stack
        # assert set(original_stack.dims) == set(ds2.dims), "Dimension names don't match between the two datasets."
    
        # 4. RMSE
        # overall rmse
        rmse = _rmse(computed_stack, original_stack, normalized)
        
        if segment is False:
            out = {'rmse': rmse}
            return xr.Dataset(out)      
        
        elif segment is True: 
            # segmented rmse  
            sos = LSP_stack['sos'] # .expand_dims({'doy': sdoys})
            pos = LSP_stack['pos']
            eos = LSP_stack['eos']

            x_ = len(original_stack.coords['x'])
            y_ = len(original_stack.coords['y'])
            temp_ = xr.DataArray(data = np.repeat(sdoys, x_*y_).reshape(len(sdoys), y_, x_),
                                    dims = original_stack.dims, 
                                    coords = original_stack.coords,
                                    attrs = original_stack.attrs).chunk(computed_stack.chunks)

            sosm = computed_stack.where(temp_ >= sos)
            eosm = computed_stack.where(temp_ <= eos)
            posm = computed_stack.where((temp_ > sos) & (temp_ < eos))

            rmse_sos = _rmse(sosm, original_stack, normalized)
            rmse_pos = _rmse(posm, original_stack, normalized)
            rmse_eos = _rmse(eosm, original_stack, normalized)

            # TODO: take into account the option of a persist option (to persist here).
            out = {'rmse': rmse, 
                    'rmse_sos': rmse_sos,
                    'rmse_pos': rmse_pos,
                    'rmse_eos': rmse_eos}              
            return xr.Dataset(out)
        else:
            raise ValueError("segment should be either True or False")
    
    def get_timeseries_metrics(self, window_length=3, metric="all", interpolType='linear', 
                            rollWindow=5, nGS=52, RMSEnormalized=True, nan_replace=None, 
                            interpolate_nans=False):
        """
        Calculate and return a specified phenological metric timeseries.

        Parameters:
        - ds (xarray.Dataset): 
            Input dataset containing the data to process.
        - years_list (list): 
            List of years to consider for timeseries extraction.
        - rollWindow (int): 
            Rolling window size used in the PhenoShape calculation.
        - metric (str, list): 
            The specific metric to return. Choices are 'sos', 'pos', 'eos', 'los', 'msp', 'mau', 'vmsp', 
                        'vmau', 'aos', 'ios', 'rog', 'ros', 'rmse' [only overall rmse], 'curvature', or 'all'. Default is 'all'.
        - nGS: Integer
            Number of observations to predict the PhenoShape
            default is 46; one per week
        - interpolate_nans: boolean
            should NaNs values be interpolated?

        Returns:
        - xarray.DataArray: A timeseries of the specified metric.
        - list of xarray.DataArray: A list of timeseries for each metric.

        Usage:
        - To get the SOS timeseries: get_timeseries(ds, window_length=3, rollWindow=5, metric="sos")
        - To get the RMSE timeseries: get_timeseries(ds, window_length=3, rollWindow=5, metric=["eos", "rmse"])
        """

        metrics_dict = {
            "sos": [],
            "pos": [],
            "eos": [],
            "vsos": [],
            "vpos": [],
            "veos": [],
            "los": [],
            "msp": [],
            "mau": [],
            "vmsp": [],
            "vmau": [],
            "ampl": [],
            "ios": [],
            "rog": [],
            "ros": [],
            "sw": [],
            #"rmse": [],
            "curvature": []
        }

        ds = self._obj

        if 'rmse' in metric or 'all' in metric:
            metrics_dict["rmse"] = []
            
        # get a list of years from the dataset
        if window_length <= 0:
            raise ValueError("Window length should be a positive integer.")
        years = np.unique(ds.year.values)
        years_list = [tuple(years[i:i+window_length]) for i in range(len(years) - window_length + 1)]

        # loop through the list of years and calculate the phenoshape, LSP, RMSE, and Curvature for each timeWindow
        for i in range(len(years_list)):
            sample = ds.where(ds.year.isin(years_list[i]), drop=True)
            year_sample = sample.year.values
            mean_year = np.mean(year_sample).astype(int)

            # Ensure the dataset is chunked properly along the 'time' dimension
            if np.unique(sample.year).size < 2:
                print("Error: Time windows should have more than one year")

            phenoshape = sample.pheno.PhenoShape(interpolType=interpolType, nan_replace=nan_replace, rollWindow=rollWindow, nGS=nGS)
            lsp = phenoshape.pheno.PhenoLSP(nGS=nGS)
            lsp = lsp.assign_coords(year=mean_year)
            #rmse_val = phenoshape.pheno.RMSE(ds, LSP_stack=lsp, normalized=RMSEnormalized, nan_replace=nan_replace, interpolate_nans=interpolate_nans)
            #rmse_val = rmse_val.assign_coords(year=mean_year)
            # Only estimate rmse if it's provided in the metric list
            if 'rmse' in metric or 'all' in metric:
                try:
                    rmse_val = phenoshape.pheno.RMSE(ds, LSP_stack=lsp, normalized=RMSEnormalized, nan_replace=nan_replace, interpolate_nans=interpolate_nans)
                    rmse_val = rmse_val.assign_coords(year=mean_year)
                    metrics_dict["rmse"].append(rmse_val.rmse)
                except:
                    print("Failed to compute RMSE for year:", mean_year)
                    rmse_val = None  # Set to None or some other placeholder value
            curvature_val = get_curvature(phenoshape)
            curvature_val = curvature_val.assign_coords(year=mean_year)

            # append each metric to the corresponding list in the dictionary
            metrics_dict["sos"].append(lsp.sos)
            metrics_dict["pos"].append(lsp.pos)
            metrics_dict["eos"].append(lsp.eos)
            metrics_dict["vsos"].append(lsp.vsos)
            metrics_dict["vpos"].append(lsp.vpos)
            metrics_dict["veos"].append(lsp.veos)
            metrics_dict["los"].append(lsp.los)
            metrics_dict["msp"].append(lsp.msp)
            metrics_dict["mau"].append(lsp.mau)
            metrics_dict["vmsp"].append(lsp.vmsp)
            metrics_dict["vmau"].append(lsp.vmau)
            metrics_dict["ampl"].append(lsp.ampl)
            metrics_dict["ios"].append(lsp.ios)
            metrics_dict["rog"].append(lsp.rog)
            metrics_dict["ros"].append(lsp.ros)
            metrics_dict["sw"].append(lsp.sw)
            metrics_dict["curvature"].append(curvature_val)
    
        # concatenate the lists of each metric into a single xarray DataArray
        for key in metrics_dict:
            metrics_dict[key] = xr.concat(metrics_dict[key], dim='time')

        if metric == 'all':
            metric = list(metrics_dict.keys())

        # if a single metric string is provided, convert it to a list
        elif isinstance(metric, str):
            metric = [metric]
        
        # Return the timeseries for the specified metric(s)
        return {m: metrics_dict[m] for m in metric}


        
