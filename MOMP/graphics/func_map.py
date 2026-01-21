import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from momp.params.region_def import polygon_boundary, add_polygon
from momp.utils.land_mask import get_india_outline
from momp.graphics.maps import calculate_cmz_averages
from momp.utils.printing import tuple_to_str_range
import cartopy.crs as ccrs
from matplotlib import colors as mcolors

from momp.utils.visual import cbar_season, set_basemap
from momp.utils.land_mask import shp_outline, shp_mask


def spatial_metrics_map(da, model_name, var_name, *, years, shpfile_dir, polygon, dir_fig, region, 
                    fig=None, ax=None, figsize=(8, 6), cmap='YlOrRd', n_colors=6, 
                    cbar_ssn=False, domain_mask=False,
                    vmin=0, vmax=15, show_ylabel=True, title=None, **kwargs):
    """
    Plot spatial maps of climatology onset day of year
    """

    # Get coordinates
    lats = da.lat.values
    lons = da.lon.values
    
    # Detect resolution from latitude spacing
    lat_diff = abs(lats[1] - lats[0])
    print(f"Detected resolution: {lat_diff:.1f} degrees")

    # Plot parameters
    map_lw = 0.75
    polygon_lw = 1.25
    panel_linewidth = 0.5
    tick_length = 3
    tick_width = 0.8
    if abs(lat_diff - 2.0) < 0.1:
        txt_fsize = 8
    elif abs(lat_diff - 4.0) < 0.1:
        txt_fsize = 10
    elif abs(lat_diff - 1.0) < 0.1:
        txt_fsize = 6
    else:
        txt_fsize = 8
    
    # Define colormap levels 
    vmin, vmax = da.quantile([0.05, 0.95], dim=None, skipna=True).values
    #levels = np.arange(vmin, vmax, (vmax-vmin)/10)
    levels = linspace(vmin, vmax, 10)

    cmap_discrete = plt.cm.get_cmap(cmap, n_colors)
    if cbar_ssn:
        cmap_jjas, norm_jjas, bounds = cbar_season()
    elif n_colors > 0:
        # Use a colormap (RdYlBu_r or similar) also 'RdYlGn_r', 'Spectral_r', or 'coolwarm'
        cmap_jjas = cmap_discrete #plt.cm.Spectral
        norm_jjas = mcolors.BoundaryNorm(levels, cmap_jjas.N, extend='max')
    else:
        cmap_jjas = cmap #plt.cm.Spectral
        norm_jjas = mcolors.Normalize(vmin=vmin, vmax=vmax)  # set explicit min/max

    # -----------------------------------------------------------------------------
    # create figure obj and ax
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    if ax is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # set map extent, country boundary, gridline
    ax, gl = set_basemap(ax, region, shpfile_dir, polygon, **kwargs)

#    # this block doesn't work since set_basemap use gl
#    ax.grid(False)
#    ax.set_axisbelow(False)
#    ax.tick_params('both', length=tick_length, width=tick_width, which='major')
#    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
#    ax.tick_params(axis='y', which='minor', left=False, right=False)

    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 
    if polygon:
        ax, polygon1_lat, polygon1_lon, polygon_defined = \
                                    add_polygon(ax, da, polygon, return_polygon=True)

    # Calculate statistics (only calculate CMZ stats if polygon is defined)
    if polygon_defined:
        cmz_onset_mean = calculate_cmz_averages(da, polygon1_lon, polygon1_lat)
    else:
        cmz_onset_mean = np.nan

    # mask data inside country boundary
    if domain_mask:
        da = shp_mask(da, region=region)


    # Plot discrete values at each grid point using pcolormesh with custom colormap
    im = ax.pcolormesh(da.lon, da.lat, da.values, 
                     cmap=cmap_jjas, norm=norm_jjas, transform=ccrs.PlateCarree(), shading='auto')
    

    # Add colorbar with MMM DD labels for every other tick
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=20)
    
    if cbar_ssn:
        # Create tick positions - use every other bound for labeling
        tick_positions = bounds[::2]  # Every other boundary
        tick_labels = [doy_to_mmm_dd(doy) for doy in tick_positions[:-1]]  # Exclude last boundary
        
        # Set all boundaries as minor ticks (for visual separation)
        cbar.set_ticks(bounds, minor=True)
        # Set every other boundary as major ticks (with labels)
        cbar.set_ticks(tick_positions[:-1])  # Exclude last boundary
    else:
        # Create custom tick labels in MMM DD format
        tick_levels = levels[::4]  # Use every other level to avoid crowding
        tick_labels = [doy_to_mmm_dd(doy) for doy in tick_levels]
        cbar.set_ticks(tick_levels)

    cbar.set_ticklabels(tick_labels)
    
    cbar.set_label(var_name, fontsize=12, fontweight='normal')
    cbar.ax.tick_params(labelsize=10)    

    # Add model name text in top-right
    ax.text(0.95, 0.95, model_name, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top',
            color='black', fontsize=15, fontweight='bold')

    # Add text annotations for onset days
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = da.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > 200 else 'black'
                ax.text(lon, lat, f'{value:.1f}', 
                           ha='center', va='center',
                           color=text_color, fontsize=txt_fsize, fontweight='normal')
    
    # Add CMZ average text (only if polygon is defined)
    if polygon_defined and not np.isnan(cmz_onset_mean):
        cmz_text = f'mean onset: {cmz_onset_mean:.0f} days'
        ax.text(0.98, 0.02, cmz_text, transform=ax.transAxes,
                    color='black', fontsize=14,
                    verticalalignment='bottom', horizontalalignment='right')

    ax.text(0.98, 0.98, 'onset (day of year)', transform=ax.transAxes,
                color='black', fontsize=14, fontweight='normal',
                verticalalignment='top', horizontalalignment='right')


    if show_ylabel:
        gl.left_labels = False

    if title:
        ax.text(0.02, 1.02, title, transform=ax.transAxes,
                verticalalignment='bottom', fontsize=15, fontweight='normal')
    
    plt.tight_layout()
    
    # Save if path provided
    if dir_fig:
        plot_filename = f"climatology_onset_{tuple_to_str_range(years_clim)}.png"
        plot_path = os.path.join(dir_fig, plot_filename)
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to: {plot_path}")
    
    plt.show()
    
    return fig, ax


def doy_to_date_string(doy, date_filter_year=2024):
    """Convert day of year to dd/mm format"""
    # Assuming non-leap year for consistency
    date = datetime(date_filter_year, 1, 1) + timedelta(days=int(doy) - 1)
    return date.strftime('%d/%m')

def doy_to_mmm_dd(doy, date_filter_year=2024):
    """Convert day of year to 'MMM DD' format"""
    # Use 2018 (not a leap year) as reference to handle all possible DOYs
    date = pd.to_datetime(f"{date_filter_year}-{int(doy):03d}", format="%Y-%j")
    return date.strftime("%b %d")


if __name__ == "__main__":
    from itertools import product
    from momp.stats.benchmark import compute_metrics_multiple_years
    from momp.lib.control import iter_list, make_case
    from momp.lib.convention import Case
    from momp.lib.loader import get_cfg, get_setting
    from dataclasses import asdict
    #from momp.graphics.onset_map import plot_spatial_climatology_onset

    cfg, setting = get_cfg(), get_setting()
    
    cfg['ref_model'] = 'climatology'
    cfg['probabilistic'] = False

    cfg_ref = cfg
    cfg_ref['model_list'] = (cfg['ref_model'],)
    #print("cfg_ref['model_list'] = ", cfg_ref['model_list'])
    layout_pool = iter_list(cfg_ref)
    #print("cfg_ref layout_pool = ", layout_pool)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, cfg_ref)
        print(f"processing model onset evaluation for {case.case_name}")

        case_ref = {'model_dir': setting.ref_model_dir,
                    'model_var': case.ref_model_var,
                    'file_pattern': setting.ref_model_file_pattern,
                    'unit_cvt': setting.ref_model_unit_cvt
                    }

        case.update(case_ref)

        if case.model == 'climatology':
            case.years = case.years_clim

        print("\n case.file_pattern = ", case.file_pattern)
        print("\n setting.ref_model_file_pattern = ", setting.ref_model_file_pattern)

        case_cfg_ref = {**asdict(case), **asdict(setting)}
        print("\n case_cfg_ref.file_pattern  = ", case_cfg_ref.get('file_pattern'))

        #print("case_cfg_ref = \n", case_cfg_ref)

        #case_cfg_ref = {**case_cfg,
        #              #'model': case_cfg['ref_model'],
        #              'model_dir': case_cfg['ref_model_dir'],
        #              'model_var': case_cfg['ref_model_var'],
        #              'file_pattern': case_cfg['ref_model_file_pattern'],
        #              'unit_cvt': case_cfg['ref_model_unit_cvt']
        #              }

        # model-obs onset benchmarking
        print("\n ", case_cfg_ref)
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)
        print("\n case_cfg_ref.file_pattern  = ", case_cfg_ref.get('years_clim'))
        break


    plot_spatial_climatology_onset(onset_da_dict, 
                                   figsize=(8, 6), cbar_ssn=False, domain_mask=False, **case_cfg_ref)

