###############################################################################
#
# PhenoPy is a Python 3.X library to process phenology indices derived from
# EarthObservation data.
#
###############################################################################

from __future__ import annotations

# standard library
import warnings

# common dependencies
import numpy as np
import xarray as xr  # manipulate 3D time-series rasters

from .curvature import get_curvature

# functions from sibling modules
from .utils import _getLSPmetrics2, _getPheno0, _rmse


@xr.register_dataarray_accessor("pheno")
class Pheno:
    def __init__(self, xr_obj: xr.DataArray) -> None:
        self._obj = xr_obj
        self.kwargs = {}
        self.LSP_bands = [
            "sos",
            "pos",
            "eos",
            "vsos",
            "vpos",
            "veos",
            "los",
            "msp",
            "mau",
            "vmsp",
            "vmau",
            "ampl",
            "ios",
            "rog",
            "ros",
            "sw",
        ]

    def PhenoShape(
        self,
        interpolType: str = "linear",
        nan_replace: float | None = None,
        rollWindow: int = 5,
        nGS: int = 52,
        recon_params: dict | None = None,
        chunks: dict | None = None,
    ) -> xr.DataArray:
        """
        Reconstruct a smoothed phenological shape per pixel.

        Maps the chosen reconstruction method along the time axis with
        ``xarray.apply_ufunc`` (Dask-parallelised when the input is chunked).

        :param interpolType: reconstruction method (see ``phenopy.reconstruction``).
        :param nan_replace: value to treat as NaN before reconstruction.
        :param rollWindow: moving-average window applied to the reconstructed curve.
        :param nGS: number of output points per cycle (default 52, ~weekly).
        :param recon_params: optional dict of method-specific parameters.
        :param chunks: optional spatial chunking (e.g. ``{"x": 300, "y": 300}``) for
            out-of-core / parallel processing; the time axis is kept whole.
        :returns: an xarray.DataArray with a ``doy`` dimension of length ``nGS``.
        """
        stack = self._obj
        doy = stack.doy.values
        xnew = np.linspace(np.min(doy), np.max(doy), nGS, dtype=np.int32)

        # drop the time-associated coords so the new 'doy' output dim can't
        # clash with the input 'doy' coordinate
        stack = stack.drop_vars(["doy", "year"], errors="ignore")
        if chunks is not None:
            stack = stack.chunk(chunks)
        if stack.chunks is not None:
            # apply_ufunc consumes the whole time axis per pixel
            stack = stack.chunk({"time": -1})

        stackP = xr.apply_ufunc(
            _getPheno0,
            stack,
            input_core_dims=[["time"]],
            output_core_dims=[["doy"]],
            exclude_dims={"time"},
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"doy": nGS}},
            output_dtypes=[float],
            kwargs={
                "doy": doy,
                "interpolType": interpolType,
                "nan_replace": nan_replace,
                "rollWindow": rollWindow,
                "nGS": nGS,
                "recon_params": recon_params,
            },
        )
        return stackP.assign_coords(doy=xnew).transpose("doy", ...)

    def PhenoLSP(
        self,
        nGS: int | None = None,
        phentype: int = 1,
        extraction: str | None = None,
        extract_params: dict | None = None,
    ) -> xr.Dataset:
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
        stack = self._obj
        if "doy" not in stack.dims:
            raise ValueError(
                "PhenoLSP expects a PhenoShape output (a 'doy' dimension). Run PhenoShape() first."
            )

        xnew = stack["doy"].values
        if nGS is None:
            nGS = len(xnew)
        n_ = len(self.LSP_bands)

        if stack.chunks is not None:
            stack = stack.chunk({"doy": -1})

        stackP = xr.apply_ufunc(
            _getLSPmetrics2,
            stack,
            input_core_dims=[["doy"]],
            output_core_dims=[["LSP_bands"]],
            exclude_dims={"doy"},
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"LSP_bands": n_}},
            output_dtypes=[float],
            kwargs={
                "xnew": xnew,
                "nGS": nGS,
                "bands": self.LSP_bands,
                "phentype": phentype,
                "extraction": extraction,
                "extract_params": extract_params,
            },
        )
        return stackP.assign_coords(LSP_bands=self.LSP_bands).to_dataset("LSP_bands")

    def RMSE(
        self,
        original_stack: xr.DataArray,
        LSP_stack: xr.Dataset,
        normalized: bool = False,
        nan_replace: float | None = None,
        interpolate_nans: bool = False,
        segment: bool = False,
    ) -> xr.Dataset:
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
        computed_stack = self._obj  # inShape, phen  || # original_stack = inData = dstack

        # 1. Check this is PhenoShape output (it must carry a 'doy' dimension)
        if "doy" not in computed_stack.dims:
            raise ValueError(
                "RMSE expects a PhenoShape output (a 'doy' dimension). Run PhenoShape() first."
            )

        if nan_replace is not None:
            original_stack = original_stack.where(original_stack.values != nan_replace)

        # 2. Get day of the year of original_stack and reorder by that
        doys = original_stack.time.dt.dayofyear.values
        sdoys = sorted(doys)
        original_stack = original_stack.assign_coords(time=doys)
        if "doy" in original_stack.coords or "doy" in original_stack.dims:
            pass
        else:
            original_stack = original_stack.rename({"time": "doy"})

        original_stack = original_stack.sortby("doy")

        # 3. Linear interpolation for pheno, to match original_stack doys
        if interpolate_nans:
            computed_stack = computed_stack.interpolate_na("doy")

        computed_stack = computed_stack.interp(doy=sdoys, method="linear")

        # If the dimension 'time' exists and 'doy' doesn't, rename 'time' to 'doy'
        if "time" in original_stack.dims and "doy" not in original_stack.dims:
            if "doy" in original_stack.coords:
                original_stack = original_stack.drop_vars("doy")
            original_stack = original_stack.rename({"time": "doy"})

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
            out = {"rmse": rmse}
            return xr.Dataset(out)

        elif segment is True:
            # segmented rmse
            sos = LSP_stack["sos"]  # .expand_dims({'doy': sdoys})
            eos = LSP_stack["eos"]

            x_ = len(original_stack.coords["x"])
            y_ = len(original_stack.coords["y"])
            temp_ = xr.DataArray(
                data=np.repeat(sdoys, x_ * y_).reshape(len(sdoys), y_, x_),
                dims=original_stack.dims,
                coords=original_stack.coords,
                attrs=original_stack.attrs,
            ).chunk(computed_stack.chunks)

            sosm = computed_stack.where(temp_ >= sos)
            eosm = computed_stack.where(temp_ <= eos)
            posm = computed_stack.where((temp_ > sos) & (temp_ < eos))

            rmse_sos = _rmse(sosm, original_stack, normalized)
            rmse_pos = _rmse(posm, original_stack, normalized)
            rmse_eos = _rmse(eosm, original_stack, normalized)

            # TODO: take into account the option of a persist option (to persist here).
            out = {"rmse": rmse, "rmse_sos": rmse_sos, "rmse_pos": rmse_pos, "rmse_eos": rmse_eos}
            return xr.Dataset(out)
        else:
            raise ValueError("segment should be either True or False")

    def get_timeseries_metrics(
        self,
        window_length: int = 3,
        metric: str | list[str] = "all",
        interpolType: str = "linear",
        rollWindow: int = 5,
        nGS: int = 52,
        RMSEnormalized: bool = True,
        nan_replace: float | None = None,
        interpolate_nans: bool = False,
        recon_params: dict | None = None,
        extraction: str | None = None,
        extract_params: dict | None = None,
    ) -> dict:
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
            # "rmse": [],
            "curvature": [],
        }

        ds = self._obj

        if "rmse" in metric or "all" in metric:
            metrics_dict["rmse"] = []

        # get a list of years from the dataset
        if window_length <= 0:
            raise ValueError("Window length should be a positive integer.")
        years = np.unique(ds.year.values)
        years_list = [
            tuple(years[i : i + window_length]) for i in range(len(years) - window_length + 1)
        ]

        # loop through the list of years and calculate the phenoshape, LSP, RMSE, and Curvature for each timeWindow
        for i in range(len(years_list)):
            sample = ds.where(ds.year.isin(years_list[i]), drop=True)
            year_sample = sample.year.values
            mean_year = np.mean(year_sample).astype(int)
            phenoshape = sample.pheno.PhenoShape(
                interpolType=interpolType,
                nan_replace=nan_replace,
                rollWindow=rollWindow,
                nGS=nGS,
                recon_params=recon_params,
            )
            lsp = phenoshape.pheno.PhenoLSP(
                nGS=nGS, extraction=extraction, extract_params=extract_params
            )
            lsp = lsp.assign_coords(year=mean_year)
            # rmse_val = phenoshape.pheno.RMSE(ds, LSP_stack=lsp, normalized=RMSEnormalized, nan_replace=nan_replace, interpolate_nans=interpolate_nans)
            # rmse_val = rmse_val.assign_coords(year=mean_year)
            # Only estimate rmse if it's provided in the metric list
            if "rmse" in metric or "all" in metric:
                try:
                    rmse_val = phenoshape.pheno.RMSE(
                        ds,
                        LSP_stack=lsp,
                        normalized=RMSEnormalized,
                        nan_replace=nan_replace,
                        interpolate_nans=interpolate_nans,
                    )
                    rmse_val = rmse_val.assign_coords(year=mean_year)
                    metrics_dict["rmse"].append(rmse_val.rmse)
                except Exception as e:
                    warnings.warn(f"Failed to compute RMSE for year {mean_year}: {e}")
                    rmse_val = None  # placeholder
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
            metrics_dict[key] = xr.concat(metrics_dict[key], dim="time")

        if metric == "all":
            metric = list(metrics_dict.keys())

        # if a single metric string is provided, convert it to a list
        elif isinstance(metric, str):
            metric = [metric]

        # Return the timeseries for the specified metric(s)
        return {m: metrics_dict[m] for m in metric}
