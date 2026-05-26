import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# functions from sibling modules
from .utils import _getLSPmetrics2, _getPheno0


def display_map(x, y, crs="EPSG:4326", margin=-0.5, zoom_bias=0):
    """
    Given a set of x and y coordinates, this function generates an
    interactive map with a bounded rectangle overlayed on Google Maps
    imagery. Based on DEAfrica notebooks

    Last modified: September 2019

    Modified from function written by Otto Wagner available here:
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities

    Parameters
    ----------
    x : (float, float)
        A tuple of x coordinates in (min, max) format.
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates.
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad
        the edges of the rectangular overlay polygon. A larger value
        results more space between the edge of the plot and the sides
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom
        level by one step. Defaults to 0; set to greater than 0 to zoom
        in, and less than 0 to zoom out.

    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A
    rectangle is drawn on this map detailing the perimeter of the x, y
    bounds.  A zoom level is calculated such that the resulting
    viewport is the closest it can possibly get to the centered
    bounding rectangle without clipping it.
    """

    def _degree_to_zoom_level(l1, l2, margin=0.0):
        """
        Helper function to set zoom level for `display_map`
        """
        degree = abs(l1 - l2) * (1 + margin)
        zoom_level_int = 0
        if degree != 0:
            zoom_level_float = math.log(360 / degree) / math.log(2)
            zoom_level_int = int(zoom_level_float)
        else:
            zoom_level_int = 18
        return zoom_level_int

    # heavy optional plotting deps imported lazily (the [plot] extra)
    import folium
    from pyproj import Transformer

    # Convert each corner coordinate to lat-lon (modern pyproj>=2 API;
    # always_xy keeps (lon/easting, lat/northing) order on input and output)
    all_x = (x[0], x[1], x[0], x[1])
    all_y = (y[0], y[0], y[1], y[1])
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    all_longitude, all_latitude = transformer.transform(all_x, all_y)

    # Calculate zoom level based on coordinates
    lat_zoom_level = (
        _degree_to_zoom_level(min(all_latitude), max(all_latitude), margin=margin) + zoom_bias
    )
    lon_zoom_level = (
        _degree_to_zoom_level(min(all_longitude), max(all_longitude), margin=margin) + zoom_bias
    )
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google",
    )

    # Create bounding box coordinates to overlay on map
    line_segments = [
        (all_latitude[0], all_longitude[0]),
        (all_latitude[1], all_longitude[1]),
        (all_latitude[3], all_longitude[3]),
        (all_latitude[2], all_longitude[2]),
        (all_latitude[0], all_longitude[0]),
    ]

    # Add bounding box as an overlay
    interactive_map.add_child(
        folium.features.PolyLine(locations=line_segments, color="red", opacity=0.8)
    )

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map


def PhenoPlot(
    stack,
    X,
    Y,
    interpolType="linear",
    saveFigure=None,
    ylim=None,
    rollWindow=None,
    nan_replace=None,
    correctionValue=None,
    plotType=1,
    phentype=1,
    nGS=52,
    fontsize=14,
    titlesize=15,
    legendsize=15,
    labelsize=13,
    threshold=300,
    ylab="NDVI",
    ax=None,
    southern=False,
    cmap="viridis",
    many_years=8,
):
    """
    Plot the PhenoShape curve along with the yearly data

    Parameters
    ----------
    - X: Float
            X coordinates
    - Y: Float
            Y coordinates
    - inData: String
            Absolute path to the original timeseries data
    - interpolType = String or Integer
            Interpolation type. Must be a string of ‘linear’, ‘nearest’,
            ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘RBF‘, ‘previous’,
            ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer
            to a spline interpolation of zeroth, first, second or third order;
            ‘previous’ and ‘next’ simply return the previous or next value"
            of the point) or as an integer specifying the order of the"
            spline interpolator to use. RBF uses cubic interpolation.
            Default is ‘linear’.
    - saveFigure: String
            Absolute path with extention to save figure on disk
    - ylim: List of Integers or Float
            Limits of the Y axis [default the y min() and max() values]
    - plotType: Type of plot, where 1 = plot with accumulated years; 2 = plot with
            start of the season (SOS), peak of the season (POS) and end of
            season (EOS);
            default is 1
    - phenType: Type os estimation of SOS and EOS. 1 = median value between POS and start and end of season. 2 = using the knee inflexion method.
            default 1
    - rollWindow: Integers
            Value of avarage smoothing of linear trend [default None]
    - nGS: Integer
            Number of observations to predict the PhenoShape
            default is 46; one per week
    - ylab: string
            Label of the Y axis [default "NDVI"]
    - southern: bool
            Reorder/relabel the x-axis to Southern-Hemisphere day-of-year so the
            austral growing season is centred [default False]
    - cmap: string
            Colormap used when there are many years [default "viridis"]
    - many_years: int
            Above this many years, points are coloured by a continuous palette
            with a colorbar instead of a per-year legend [default 8]

    """

    # Get the per-pixel time series
    doy = stack.doy.values
    dates = stack.time
    valuesTSS = stack.sel(x=X, y=Y, method="nearest").values

    valuesTSSpd = pd.DataFrame(
        {"doy": dates.dt.dayofyear, "year": dates.dt.year, "VI": valuesTSS}
    ).sort_values("doy")

    # Southern Hemisphere: map calendar DOY -> day-of-season (austral year ~1 July)
    if southern:
        cd = valuesTSSpd["doy"].values
        plot_doy = np.where(cd >= 183, cd - 183, cd + 182)
        fit_doy = np.where(doy >= 183, doy - 183, doy + 182)
        xlabel = "Day of season (Southern Hemisphere)"
    else:
        plot_doy = valuesTSSpd["doy"].values
        fit_doy = doy
        xlabel = "Day of the year"
    valuesTSSpd["pdoy"] = plot_doy

    phen = _getPheno0(
        y=valuesTSS,
        doy=fit_doy,
        interpolType=interpolType,
        nan_replace=nan_replace,
        rollWindow=rollWindow,
        nGS=nGS,
    )
    xnew = np.linspace(np.min(plot_doy), np.max(plot_doy), nGS, dtype="int16")

    if ax is None:
        _, ax = plt.subplots()

    if plotType == 1:
        years = sorted(valuesTSSpd["year"].unique())
        if len(years) > many_years:
            # many years -> continuous palette + colorbar (instead of a large legend)
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(min(years), max(years)), cmap=cmap)
            sm.set_array([])
            for name, group in valuesTSSpd.groupby("year"):
                ax.plot(group["pdoy"], group["VI"], "o", ms=5, color=sm.to_rgba(name))
            ax.figure.colorbar(sm, ax=ax, pad=0.01).set_label("year", fontsize=fontsize)
        else:
            for name, group in valuesTSSpd.groupby("year"):
                ax.plot(group["pdoy"], group["VI"], "o", ms=8, label=int(name))
            # legend outside the plot, on the left
            ax.legend(
                loc="center right",
                bbox_to_anchor=(-0.12, 0.5),
                fontsize=legendsize,
                title="year",
                frameon=False,
            )
        ax.plot(xnew, phen, "-", color="black", lw=2)
        ax.tick_params(labelsize=labelsize)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if southern:
            _relabel_southern(ax)

    elif plotType == 2:
        # get position of SOS, POS, and EOS
        metrics = _getLSPmetrics2(phen, xnew, nGS, len(xnew), phentype)
        isos = np.where(xnew == metrics[0])[0][0]
        ipos = np.where(xnew == metrics[1])[0][0]
        ieos = np.where(xnew == metrics[2])[0][0]
        ax.plot(xnew, phen, "-", color="black")
        ax.plot(xnew[isos], phen[isos], "X", markersize=15, label="SOS")
        ax.plot(xnew[ipos], phen[ipos], "X", markersize=15, label="POS")
        ax.plot(xnew[ieos], phen[ieos], "X", markersize=15, label="EOS")
        ax.legend(
            loc="center right", bbox_to_anchor=(-0.12, 0.5), fontsize=legendsize, frameon=False
        )
        ax.tick_params(labelsize=labelsize)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        if southern:
            _relabel_southern(ax)

    if saveFigure is not None:
        ax.figure.savefig(saveFigure, bbox_inches="tight")

    return ax


def _relabel_southern(ax):
    """Relabel x-ticks (day-of-season positions) as real Southern-Hemisphere
    calendar DOY (position ``p`` -> ``((p + 182) % 365) + 1``), keeping the
    data-driven axis limits (``set_xticks`` would otherwise expand the view to
    the locator's out-of-range ticks)."""
    xlim = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([int(((t + 182) % 365) + 1) for t in ticks])
    ax.set_xlim(xlim)


def plot_with_southern_doy(shape, coordinates, ylabel="NDVI", title=None):
    """
    Plot the data from a specified shape with x-ticks reordered to represent real
    southern hemispherical day-of-the-year (DOY) values.

    Parameters:
    - shape: xarray.DataArray
        The data shape to plot.
    - coordinates: tuple
        The coordinates of the pixel to plot.
    - ylabel: str, optional
        vegetation index used in the analysis. Default is'NDVI
    - title: str, optional
        The title for the plot. Default is None, meaning no title.

    Returns:
    - ax: matplotlib.axes._subplots.AxesSubplot
        The plotted axis.
    """

    # Plot the data with real southern hemispherical doys
    X = coordinates[0]
    Y = coordinates[1]
    shape.sel(x=X, y=Y, method="nearest").plot()

    # Fetch the current active axis using Matplotlib's gca
    ax = plt.gca()

    # Get the original x-ticks
    original_ticks = ax.get_xticks()

    # Generate reordered labels based on the original ticks
    doy1 = np.linspace(183, 365, len(original_ticks) // 2, dtype=int)
    doy2 = np.linspace(1, 182, len(original_ticks) // 2, dtype=int)
    doy3 = np.concatenate((doy1, doy2))

    # Set the new x-tick labels
    ax.set_xticks(original_ticks)
    ax.set_xticklabels(doy3)

    # If a title is provided, set it
    if title:
        plt.title(title)
    # add y label
    plt.ylabel(ylabel)

    return ax
