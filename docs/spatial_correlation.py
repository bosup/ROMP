import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from pykrige.ok import OrdinaryKriging
from matplotlib.path import Path
import cartopy.crs as ccrs


from momp.utils.land_mask import get_india_outline, shp_mask


# helpers
def h_to_km(h, grid_res_deg=0.25):
    """
    Convert a lag distance in grid cells to kilometres.

    Parameters
    ----------
    h : int or float
        Lag distance in grid cells.
    grid_res_deg : float, optional
        Spatial resolution of the grid in degrees. Default is 0.25°.

    Returns
    -------
    float
        Approximate distance in kilometres, assuming 111.32 km per degree.
    """
    return h * grid_res_deg * 111.32


def load_data_year_set(path_to_data,
                       start_year,
                       end_year,
                       file_pattern="/{}.nc",
                       just_et=False
                       ):
    """
    Load and concatenate yearly NetCDF files into a single DataArray.

    Iterates over each year in [start_year, end_year], opens the corresponding
    NetCDF file, and concatenates all arrays along the TIME dimension.
    Optionally masks the result to the Ethiopia region.

    Parameters
    ----------
    path_to_data : str
        Root directory containing the yearly .nc files.
    start_year : int
        First year (inclusive) to load.
    end_year : int
        Last year (inclusive) to load.
    file_pattern : str, optional
        File naming pattern with a single ``{}`` placeholder for the year.
        Default is ``"/{}.nc"``.
    just_et : bool, optional
        If True, rename coordinate dimensions from LONGITUDE/LATITUDE to
        lon/lat and apply the Ethiopia shapefile mask. Default is False.

    Returns
    -------
    xarray.DataArray
        Combined DataArray concatenated along the TIME dimension.

    Raises
    ------
    ValueError
        If ``start_year`` is greater than ``end_year``.
    """
    if start_year > end_year:
        raise ValueError("start_year must be before end_year")
    
    da_list = []
    for year in range(start_year, end_year+1):
        loop_path = path_to_data + file_pattern.format(year)
        loop_da = xr.open_dataarray(loop_path)
        da_list.append(loop_da)

    combined_data = xr.concat(da_list, dim="TIME")

    if just_et:
        combined_data = combined_data.rename(
            {"LONGITUDE": "lon",
             "LATITUDE": "lat"}
        )
        combined_data = shp_mask(combined_data, region='Ethiopia')

    return combined_data


def centers_to_edges(centers):
    """
    Convert a 1-D array of grid-cell centres to cell edges.

    Assumes uniform spacing between centres. The left/bottom edge is
    extrapolated half a step below the first centre; the right/top edge is
    extrapolated half a step above the last centre.

    Parameters
    ----------
    centers : array-like of float
        1-D array of uniformly-spaced cell-centre coordinates.

    Returns
    -------
    numpy.ndarray
        Array of length ``len(centers) + 1`` containing the cell edges.
    """
    step = centers[1] - centers[0]          # assumes uniform spacing
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2   # midpoints between centers
    edges[0]    = centers[0]  - step / 2              # extrapolate left/bottom
    edges[-1]   = centers[-1] + step / 2              # extrapolate right/top
    return edges


def plot_mean_rainfall_data(da,
                            vmin=0, vmax=5,
                            shp_file_path=None,
                            title=None):
    """
    Plot a map of mean daily rainfall from a spatial DataArray.

    Computes the temporal mean across the TIME dimension and displays the
    result as a pcolormesh map using a Plate Carree projection. Optionally
    overlays regional boundaries from a shapefile.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with dimensions (TIME, lat, lon). Must have ``lon`` and
        ``lat`` coordinate arrays.
    vmin : float, optional
        Minimum value for the colorbar. Default is 0.
    vmax : float, optional
        Maximum value for the colorbar. Default is 5.
    shp_file_path : str or None, optional
        Path to a shapefile whose boundaries are drawn on the map. If None,
        no boundaries are drawn. Default is None.
    title : str or None, optional
        Title string for the plot. If None, no title is set. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes (GeoAxes)
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    lons = da.lon.values
    lats = da.lat.values

    LON_edges = centers_to_edges(lons)
    LAT_edges = centers_to_edges(lats)

    mean_rain_vals = da.mean(dim="TIME").values

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(LON_edges, LAT_edges, mean_rain_vals,
                        cmap="YlOrRd", vmin=vmin, vmax=vmax, shading='flat')

    buffer = 0.5
    ax.set_extent([
        lons.min() - buffer,
        lons.max() + buffer,
        lats.min() - buffer,
        lats.max() + buffer
    ], crs=ccrs.PlateCarree())

    if shp_file_path:
        et_boundaries = get_india_outline(shp_file_path)
        for boundary in et_boundaries:
            et_lon, et_lat = boundary
            ax.plot(et_lon, et_lat, color='black', linewidth=0.5)

    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Mean Rainfall (mm/day)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Axis labels
    gl = ax.gridlines(draw_labels=True, linewidth=0, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    plt.tight_layout()

    return fig, ax


def get_pairwise_values(h, data, vertical=False):
    """
    Extract all pairs of values separated by lag ``h`` from a 2-D array.

    Slides a window of width ``h`` along each row (or column, if
    ``vertical=True``) and records the value at the start and end of each
    window. Pairs containing NaN or Inf in either position are dropped.

    Parameters
    ----------
    h : int
        Lag distance in grid cells. Must be >= 1.
    data : numpy.ndarray
        2-D array of values.
    vertical : bool, optional
        If True, transpose the array before processing so that pairs are
        collected along columns rather than rows. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``"A"`` (start values), ``"B"`` (end values),
        and ``"h"`` (the lag, repeated for every row).

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, or if ``h`` is less than 1.
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D")
    if h < 1:
        raise ValueError("h must be greater than 0")
    if vertical:
        data = data.T

    start_vals = []
    end_vals = []

    for row in data:
        start, end = 0, h
        while end < len(row):
            s, e = row[start], row[end]
            if np.isfinite(s) and np.isfinite(e):
                start_vals.append(s)
                end_vals.append(e)
            start += 1
            end += 1
        
    ret_df = pd.DataFrame({
        "A": start_vals,
        "B": end_vals,
        "h": [h]*len(start_vals)
    })

    return ret_df


def get_dict_of_pairwise_vals(h_range: list, data):
    """
    Build a dictionary of pairwise-value DataFrames for multiple lag distances.

    For each lag ``h`` in ``h_range``, collects horizontal and vertical pairs
    from ``data`` and concatenates them into a single DataFrame.

    Parameters
    ----------
    h_range : list of int
        Sequence of lag distances (in grid cells) to process.
    data : numpy.ndarray
        2-D array of spatial values.

    Returns
    -------
    dict
        Mapping from each lag ``h`` to a :class:`pandas.DataFrame` with
        columns ``"A"``, ``"B"``, and ``"h"``, containing all finite
        horizontal and vertical pairs at that lag.
    """
    ret_dict = {}
    for h in h_range:
        hor_pairs = get_pairwise_values(h, data)
        ver_pairs = get_pairwise_values(h, data, vertical=True)
        ret_dict[h] = pd.concat([hor_pairs, ver_pairs])

    return ret_dict


def get_pairwise_diffs(h, data, vertical=False):
    """
    Compute pairwise differences (A - B) for all pairs separated by lag ``h``.

    Slides a window of width ``h`` along each row (or column, if
    ``vertical=True``) and records the difference between the start and end
    values. Pairs where either value is non-finite are skipped.

    Parameters
    ----------
    h : int
        Lag distance in grid cells. Must be >= 1.
    data : numpy.ndarray
        2-D array of values.
    vertical : bool, optional
        If True, transpose the array so that differences are computed along
        columns. Default is False.

    Returns
    -------
    list of float
        Differences (start − end) for all valid pairs at lag ``h``.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, or if ``h`` is less than 1.
    """
    if data.ndim != 2:
        raise ValueError("Data must be 2D")
    if h < 1:
        raise ValueError("h must be greater than 0")
    if vertical:
        data = data.T
    
    pair_diffs = []
    for row in data:
        start, end = 0, h
        while end < len(row):
            s, e = row[start], row[end]
            if np.isfinite(s) and np.isfinite(e):
                pair_diffs.append(s - e)
            start += 1
            end += 1
    
    return pair_diffs


def get_dict_of_pairwise_diffs(h_range: list, data):
    """
    Build a dictionary of pairwise-difference arrays for multiple lag distances.

    For each lag ``h`` in ``h_range``, collects horizontal and vertical
    pairwise differences and concatenates them into a single NumPy array.

    Parameters
    ----------
    h_range : list of int
        Sequence of lag distances (in grid cells) to process.
    data : numpy.ndarray
        2-D array of spatial values.

    Returns
    -------
    dict
        Mapping from each lag ``h`` to a :class:`numpy.ndarray` of all finite
        pairwise differences (horizontal + vertical) at that lag.
    """
    ret_dict = {}
    for h in h_range:
        hor_pairs = get_pairwise_diffs(h, data)
        ver_pairs = get_pairwise_diffs(h, data, vertical=True)
        ret_dict[h] = np.array(hor_pairs + ver_pairs)

    return ret_dict


def get_corr_vals(df, col1="A", col2="B"):
    """
    Compute the Pearson correlation coefficient and p-value for two columns.

    Uses :func:`scipy.stats.linregress` on ``df[col1]`` and ``df[col2]``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the two columns to correlate.
    col1 : str, optional
        Name of the first column. Default is ``"A"``.
    col2 : str, optional
        Name of the second column. Default is ``"B"``.

    Returns
    -------
    rvalue : float
        Pearson correlation coefficient.
    pvalue : float
        Two-sided p-value for the null hypothesis that the slope is zero.
    """
    lin = linregress(df[col1], df[col2])
    return lin.rvalue, lin.pvalue


def get_multiple_corr_vals(dict_of_dfs):
    """
    Compute correlation coefficients and p-values for a range of lag distances.

    Iterates over the dictionary produced by :func:`get_dict_of_pairwise_vals`
    and calls :func:`get_corr_vals` for each lag.

    Parameters
    ----------
    dict_of_dfs : dict
        Mapping from lag ``h`` to a DataFrame with columns ``"A"`` and ``"B"``,
        as returned by :func:`get_dict_of_pairwise_vals`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        * ``"h"`` – lag values (keys of the input dict).
        * ``"rvals"`` – Pearson r for each lag.
        * ``"pvals"`` – corresponding two-sided p-values.
    """
    rvals = []
    pvals = []
    for key, val in dict_of_dfs.items():
        rv, pv = get_corr_vals(val)
        rvals.append(rv)
        pvals.append(pv)

    ret_df = pd.DataFrame({
        "h": dict_of_dfs.keys(),
        "rvals": rvals,
        "pvals": pvals
    })
    return ret_df


def plot_spatial_correlogram(df, x='h', y='rvals', p='pvals', alpha=0.01, grid_res_deg=0.25):
    """
    Plot a spatial correlogram of Pearson r against distance.

    Converts lag values from grid cells to kilometres and renders a line plot
    with scatter markers. Statistically significant points (p < alpha) are
    filled; non-significant points have open markers.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns for lag (``x``), correlation (``y``), and
        p-values (``p``), as returned by :func:`get_multiple_corr_vals`.
    x : str, optional
        Column name for the lag (in grid cells). Default is ``"h"``.
    y : str, optional
        Column name for the r-values. Default is ``"rvals"``.
    p : str, optional
        Column name for the p-values. Default is ``"pvals"``.
    alpha : float, optional
        Significance level threshold. Default is 0.01.
    grid_res_deg : float, optional
        Grid resolution in degrees, used to convert lags to km. Default is 0.25.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x_km = df[x] * grid_res_deg * 111.32
    significant = df[p] < alpha
    
    ax.plot(x_km, df[y], color='steelblue', linewidth=1, alpha=0.5)
    ax.scatter(x_km[significant], df[y][significant],
               color='steelblue', zorder=3, label=f'p < {alpha}')
    ax.scatter(x_km[~significant], df[y][~significant],
               color='steelblue', zorder=3, facecolors='none',
               edgecolors='steelblue', linewidths=1.5, label=f'p ≥ {alpha}')

    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel("r-value", fontsize=12)
    ax.set_title('Spatial Correlogram', fontsize=13)
    ax.set_xticks(x_km)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig, ax


def compute_and_plot_variogram(pairwise_dict, grid_res_deg=0.25):
    """
    Compute the empirical semivariogram and plot it.

    For each lag ``h`` in ``pairwise_dict``, computes the classical estimator::

        γ(h) = 0.5 * mean((A - B)²)

    after dropping NaN pairs, then plots semivariance against distance (km).
    The number of contributing pairs is annotated above each point.

    Parameters
    ----------
    pairwise_dict : dict
        Mapping from lag ``h`` to a DataFrame with columns ``"A"`` and ``"B"``,
        as returned by :func:`get_dict_of_pairwise_vals`.
    grid_res_deg : float, optional
        Grid resolution in degrees, used to convert lags to km. Default is 0.25.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    variogram_df : pandas.DataFrame
        DataFrame with columns ``"h"``, ``"h_km"``, ``"gamma"``, and
        ``"n_pairs"`` for each lag.
    """
    records = []
    for h, df in pairwise_dict.items():
        df_clean = df.dropna(subset=['A', 'B'])
        gamma = 0.5 * ((df_clean['A'] - df_clean['B']) ** 2).mean()
        records.append({'h': h, 
                        'h_km': h * grid_res_deg * 111.32,
                        'gamma': gamma, 
                        'n_pairs': len(df_clean)})
    
    variogram_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.scatter(variogram_df['h_km'], variogram_df['gamma'], color='firebrick', zorder=3)
    ax.plot(variogram_df['h_km'], variogram_df['gamma'], color='firebrick', linewidth=1, alpha=0.5)
    
    for _, row in variogram_df.iterrows():
        ax.annotate(f"{int(row['n_pairs'])}",
                    xy=(row['h_km'], row['gamma']),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8, color='gray')
    
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Semivariance γ(h)', fontsize=12)
    ax.set_title('Empirical Variogram', fontsize=13)
    ax.set_ylim(variogram_df['gamma'].min() * 0.9,
                variogram_df['gamma'].max() * 1.15)
    
    plt.tight_layout()
    return fig, ax, variogram_df


def detrend_and_plot_variogram(h_range, da):
    """
    Remove a linear lon/lat trend from a DataArray and plot the residual variogram.

    Fits a first-order (planar) trend surface to the temporal mean field using
    ordinary least squares, subtracts the trend to obtain residuals, then
    computes and plots the empirical semivariogram of those residuals.

    Parameters
    ----------
    h_range : list of int
        Sequence of lag distances (in grid cells) passed to
        :func:`get_dict_of_pairwise_vals`.
    da : xarray.DataArray
        DataArray with ``lon`` and ``lat`` coordinates and optionally a TIME
        dimension. The temporal mean is used if TIME is present.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    variogram_df : pandas.DataFrame
        Empirical variogram table from :func:`compute_and_plot_variogram`.
    """
    lons = da.lon
    lats = da.lat
    try:
        vals = da.mean(dim="TIME").values
    except Exception:
        vals = da.values

    lons_grid, lats_grid = np.meshgrid(lons, lats)
    flat_lons = lons_grid.ravel()
    flat_lats = lats_grid.ravel()
    flat_vals = vals.ravel()

    # Remove NaNs
    valid = np.isfinite(flat_vals)

    # Fit linear trend in lon/lat
    A = np.column_stack([flat_lons[valid], flat_lats[valid], np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, flat_vals[valid], rcond=None)

    # Build trend surface over full grid
    A_full = np.column_stack([flat_lons, flat_lats, np.ones(len(flat_lons))])
    trend = (A_full @ coeffs).reshape(vals.shape)

    # Subtract trend to get residuals
    residuals = vals - trend

    # Compute variogram on residuals
    td = get_dict_of_pairwise_vals(h_range, data=residuals)
    fig, ax, variogram_df = compute_and_plot_variogram(td)
    return fig, ax, variogram_df


def spherical(h, nugget, sill, range_):
    """
    Evaluate the spherical semivariogram model.

    The standard spherical model is::

        γ(h) = nugget + sill * [1.5*(h/a) - 0.5*(h/a)³]   for h ≤ a
        γ(h) = nugget + sill                                 for h > a

    where ``a`` is the practical range (``range_``).

    Parameters
    ----------
    h : array-like of float
        Lag distances at which to evaluate the model.
    nugget : float
        Nugget effect (semivariance at h = 0⁺).
    sill : float
        Partial sill (variance contribution of the spatially-correlated
        component).
    range_ : float
        Practical range — the lag beyond which spatial correlation becomes
        negligible.

    Returns
    -------
    numpy.ndarray
        Semivariance values corresponding to each element of ``h``.
    """
    h = np.array(h, dtype=float)
    gamma = np.where(
        h <= range_,
        nugget + sill * (1.5*(h/range_) - 0.5*(h/range_)**3),
        nugget + sill
    )
    return gamma


def plot_empirical_spherical_variogram(variogram_df, p0=None):
    """
    Fit a spherical model to an empirical variogram and plot both.

    Uses :func:`scipy.optimize.curve_fit` with a non-negative bounds
    constraint. If no initial guess is provided, data-driven defaults are
    computed from ``variogram_df``.

    Parameters
    ----------
    variogram_df : pandas.DataFrame
        Empirical variogram table with columns ``"h_km"`` and ``"gamma"``,
        as returned by :func:`compute_and_plot_variogram`.
    p0 : list of float or None, optional
        Initial guess ``[nugget, sill, range_km]`` for the curve fit.
        If None, defaults are derived automatically. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    popt : numpy.ndarray
        Optimised parameters ``[nugget, sill, range_km]``.

    Notes
    -----
    Fitted parameter values are printed to stdout.
    """
    h_vals = variogram_df['h_km'].values
    gamma_vals = variogram_df['gamma'].values

    # Data-driven defaults if no guess provided
    if p0 is None:
        p0 = [
            gamma_vals[0] * 0.5,
            gamma_vals.max() - gamma_vals.min(),
            h_vals[np.argmax(gamma_vals)] * 0.5
        ]
        print(f"Using auto initial guess — nugget: {p0[0]:.2f}, sill: {p0[1]:.2f}, range: {p0[2]:.2f}km")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(h_vals, gamma_vals, color='firebrick', zorder=3, label='Empirical')

    h_smooth = np.linspace(h_vals.min(), h_vals.max(), 200)
    popt, _ = curve_fit(spherical, h_vals, gamma_vals,
                        p0=p0,
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                        maxfev=10000)
    ax.plot(h_smooth, spherical(h_smooth, *popt),
            label=f'Spherical (nugget={popt[0]:.2f}, sill={popt[1]:.2f}, range={popt[2]:.0f}km)')
    ax.vlines(x=popt[2], ymin=min(gamma_vals), ymax=max(gamma_vals),
              color="green", linestyles="--", label="Range")

    print(f"Spherical: nugget={popt[0]:.3f}, sill={popt[1]:.3f}, range={popt[2]:.1f}km")

    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Semivariance γ(h)')
    ax.set_title('Theoretical Variogram Model - Spherical')
    plt.tight_layout()
    return fig, ax, popt


def full_spatial_analysis_pipeline(path_to_data,
                                    start_year,
                                    end_year,
                                    file_pattern="/{}.nc",
                                    h_range=range(2, 25),
                                    shp_file_path=None,
                                    title_prefix="Ethiopia",
                                    vmin=0, vmax=5,
                                    alpha=0.05):
    """
    Run the full spatial analysis pipeline for a rainfall dataset.

    Loads yearly NetCDF data and produces four diagnostic plots in sequence:

    1. **Mean rainfall map** – temporal mean displayed on a geographic map.
    2. **Spatial correlogram** – Pearson r vs. distance for the mean field.
    3. **Detrended empirical variogram** – semivariogram after removing a
       planar lon/lat trend.
    4. **Spherical model fit** – spherical variogram model overlaid on the
       empirical variogram.

    Parameters
    ----------
    path_to_data : str
        Root directory containing the yearly .nc files.
    start_year : int
        First year (inclusive) to load.
    end_year : int
        Last year (inclusive) to load.
    file_pattern : str, optional
        File naming pattern with a single ``{}`` placeholder for the year.
        Default is ``"/{}.nc"``.
    h_range : range or list of int, optional
        Lag distances (in grid cells) for the correlogram and variogram.
        Default is ``range(2, 25)``.
    shp_file_path : str or None, optional
        Path to a shapefile for map boundary overlay. Default is None.
    title_prefix : str, optional
        Prefix string prepended to each plot title. Default is ``"Ethiopia"``.
    vmin : float, optional
        Lower colorbar limit for the rainfall map. Default is 0.
    vmax : float, optional
        Upper colorbar limit for the rainfall map. Default is 5.
    alpha : float, optional
        Significance level for the correlogram scatter markers. Default is 0.05.

    Returns
    -------
    dict with keys:

    * ``"da"`` – loaded :class:`xarray.DataArray`.
    * ``"corr_df"`` – correlogram DataFrame from
      :func:`get_multiple_corr_vals`.
    * ``"variogram_df"`` – empirical variogram DataFrame.
    * ``"variogram_values"`` – optimised spherical parameters
      ``[nugget, sill, range_km]``.
    * ``"figures"`` – list of the four :class:`matplotlib.figure.Figure`
      objects in plot order.
    """
    print("=" * 50)
    print(f"Loading data: {start_year}–{end_year}")
    print("=" * 50)
    da = load_data_year_set(
        path_to_data=path_to_data,
        start_year=start_year,
        end_year=end_year,
        file_pattern=file_pattern,
        just_et=True
    )
    print(f"Loaded Data for {len(da.TIME.values)} days")
    print()

    print("=" * 50)
    print("STEP 1: Mean Rainfall Map")
    print("=" * 50)
    fig1, ax1 = plot_mean_rainfall_data(da,
                                        vmin=vmin, vmax=vmax,
                                        shp_file_path=shp_file_path,
                                        title=f'Mean Rainfall - {title_prefix}')
    plt.show()

    print("=" * 50)
    print("STEP 2: Spatial Correlogram")
    print("=" * 50)
    mean_rain_vals = da.mean(dim="TIME").values
    pairwise_dict = get_dict_of_pairwise_vals(h_range, data=mean_rain_vals)
    corr_df = get_multiple_corr_vals(pairwise_dict)
    fig2, ax2 = plot_spatial_correlogram(corr_df, alpha=alpha)
    ax2.set_title(f'Spatial Correlogram - {title_prefix}')
    plt.show()

    print("=" * 50)
    print("STEP 3: Detrended Empirical Variogram")
    print("=" * 50)
    fig3, ax3, variogram_df = detrend_and_plot_variogram(h_range, da)
    ax3.set_title(f'Empirical Variogram (Detrended) - {title_prefix}')
    plt.show()

    print("=" * 50)
    print("STEP 4: Spherical Model Fit")
    print("=" * 50)
    fig4, ax4, popt = plot_empirical_spherical_variogram(variogram_df)
    ax4.set_title(f'Spherical Variogram Model - {title_prefix}')
    plt.show()

    return {
        'da': da,
        'corr_df': corr_df,
        'variogram_df': variogram_df,
        'variogram_values': popt,
        'figures': [fig1, fig2, fig3, fig4],
    }


def create_krig(da, var_values):
    """
    Fit an Ordinary Kriging model and interpolate onto the original grid.

    Extracts all non-NaN data points from ``da``, converts their coordinates
    to kilometres, and fits a spherical Ordinary Kriging model using the
    variogram parameters supplied in ``var_values``. The model is then
    executed on a grid matching ``da``'s lon/lat coordinates exactly.

    Parameters
    ----------
    da : xarray.DataArray
        2-D spatial DataArray with ``lon`` and ``lat`` coordinate arrays.
        NaN pixels are treated as unobserved and excluded from the kriging
        input.
    var_values : array-like of length 3
        Variogram parameters in the order ``[nugget, sill, range_km]``, as
        returned by :func:`plot_empirical_spherical_variogram`.

    Returns
    -------
    krig_interp : numpy.ma.MaskedArray
        2-D kriged surface on the same grid as ``da``, with shape
        ``(len(da.lat), len(da.lon))``.
    ss : numpy.ma.MaskedArray
        2-D kriging variance (estimation uncertainty) on the same grid.

    Notes
    -----
    Coordinates are converted to km by multiplying degrees by 111.32. This
    is an equirectangular approximation and may introduce small errors at
    high latitudes or over large domains.
    """
    lon2d, lat2d = np.meshgrid(da.lon.values, da.lat.values)
    onset_days = da.values

    mask = ~np.isnan(onset_days.flatten())
    lons_krig = lon2d.flatten()[mask]
    lats_krig = lat2d.flatten()[mask]
    onset_flat = onset_days.flatten()[mask]

    # Match output grid exactly to original data
    gridx = da.lon.values
    gridy = da.lat.values

    # convert to km — note: coords themselves, not grid spacing
    lons_krig_km = lons_krig * 111.32
    lats_krig_km = lats_krig * 111.32
    gridx_km     = gridx     * 111.32
    gridy_km     = gridy     * 111.32

    n = var_values[0]
    s = var_values[1]
    r = var_values[2]

    test_ok = OrdinaryKriging(
        lons_krig_km,
        lats_krig_km,
        onset_flat,
        variogram_model="spherical",
        variogram_parameters={
            'nugget': n,
            'psill':  s,
            'range':  r
        }
    )

    krig_interp, ss = test_ok.execute("grid", gridx_km, gridy_km)

    return krig_interp, ss


def plot_krig_all(da, krig_interp, shp_file_path):
    """
    Plot the full kriged surface masked to the study region boundary.

    Builds a boolean mask from the shapefile polygons using
    :class:`matplotlib.path.Path` point-in-polygon tests, applies it to
    ``krig_interp`` so that pixels outside the boundary become NaN, then
    renders the result as a georeferenced image.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose ``lon`` and ``lat`` coordinates define the output grid.
    krig_interp : array-like
        2-D kriged surface with shape ``(len(da.lat), len(da.lon))``, as
        returned by :func:`create_krig`.
    shp_file_path : str
        Path to the shapefile used to draw regional boundaries and generate
        the land mask.

    Returns
    -------
    krig_masked : numpy.ndarray
        2-D array equal to ``krig_interp`` inside the boundary and NaN
        outside.

    Notes
    -----
    The plot is rendered to the active matplotlib display via ``plt.show()``.
    """
    # Match output grid exactly to original data
    gridx = da.lon.values
    gridy = da.lat.values

    lon_grid, lat_grid = np.meshgrid(gridx, gridy)
    mask = np.zeros(lon_grid.shape, dtype=bool)

    et_boundaries = get_india_outline(shp_file_path)
    for boundary in et_boundaries:
        et_lon, et_lat = boundary
        poly_path = Path(np.column_stack([et_lon, et_lat]))
        points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        inside = poly_path.contains_points(points).reshape(lon_grid.shape)
        mask |= inside

    krig_masked = np.where(mask, krig_interp, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    img = ax.imshow(
        krig_masked,
        extent=[gridx.min(), gridx.max(), gridy.min(), gridy.max()],
        origin='lower',
        cmap='magma',
        transform=ccrs.PlateCarree(),
        alpha=1
    )

    for boundary in et_boundaries:
        et_lon, et_lat = boundary
        ax.plot(et_lon, et_lat, color='black', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', color='grey')
    gl.top_labels = False
    gl.right_labels = False
    plt.colorbar(img, ax=ax, label='Onset Day', fraction=0.03, pad=0.08)
    plt.title('Onset Day - Kriged')
    plt.show()

    return krig_masked


def plot_krig_only_interp(da, krig_data, shp_file_path):
    """
    Plot only the pixels that were filled by kriging (originally NaN).

    Compares the original DataArray values against the kriged surface to
    identify pixels that were NaN in the input but have a finite value after
    kriging. Only those interpolated pixels are shown; all others are masked
    to NaN. The displayed values are shifted so that the minimum interpolated
    value equals zero.

    Parameters
    ----------
    da : xarray.DataArray
        Original DataArray with ``lon`` and ``lat`` coordinates. NaN pixels
        are the candidate interpolation locations.
    krig_data : array-like
        2-D kriged surface with shape ``(len(da.lat), len(da.lon))``, as
        returned by :func:`plot_krig_all` or :func:`create_krig`.
    shp_file_path : str
        Path to the shapefile used to draw regional boundary lines on the map.

    Notes
    -----
    The colorbar is fixed to the range [50, 80] with 11 evenly spaced ticks.
    The plot is rendered to the active matplotlib display via ``plt.show()``.
    No value is returned.
    """
    orig_data = da.values
    gridx = da.lon.values
    gridy = da.lat.values

    lon_grid, lat_grid = np.meshgrid(gridx, gridy)
    mask = np.zeros(lon_grid.shape, dtype=bool)

    was_nan_original = ~np.isfinite(orig_data)
    has_val_kriged   = np.isfinite(krig_data)

    interpolated = was_nan_original & has_val_kriged

    # keep only pixels that were NaN in original but filled by kriging
    interp_only = np.where(interpolated, krig_data, np.nan)
    interp_only = interp_only - np.nanmin(interp_only)

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    img = ax.imshow(
        interp_only,
        extent=[gridx.min(), gridx.max(), gridy.min(), gridy.max()],
        origin='lower',
        cmap='OrRd',
        transform=ccrs.PlateCarree(),
        vmin=50,
        vmax=80,
        alpha=1
    )

    et_boundaries = get_india_outline(shp_file_path)
    for boundary in et_boundaries:
        et_lon, et_lat = boundary
        ax.plot(et_lon, et_lat, color='black', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', color='grey')
    gl.top_labels = False
    gl.right_labels = False
    plt.colorbar(
        img,
        ax=ax,
        label='Onset Day (Normalized)',
        fraction=0.03,
        pad=0.04,
        ticks=np.linspace(50, 80, 11)
    )
    plt.title('Onset Day - Interpolated Pixels Only')
    plt.show()