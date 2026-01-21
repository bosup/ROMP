import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from momp.params.region_def import polygon_boundary
from momp.utils.land_mask import get_india_outline
from momp.graphics.maps import calculate_cmz_averages
from momp.utils.printing import tuple_to_str_range

def plot_spatial_climatology_onset(onset_da_dict, *, years_clim, shpfile_dir, polygon, dir_fig, figsize=(18, 6), **kwargs):
    """
    Plot spatial maps of climatology onset day of year
    """

    # Extract data
    climatological_onset_doy = next(iter(onset_da_dict.values()))
    
    # Get coordinates
    lats = climatological_onset_doy.lat.values
    lons = climatological_onset_doy.lon.values
    
    # Detect resolution from latitude spacing
    lat_diff = abs(lats[1] - lats[0])
    print(f"Detected resolution: {lat_diff:.1f} degrees")
    
    # Define Core Monsoon Zone bounding polygon coordinates based on resolution 
    polygon_defined = False

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(climatological_onset_doy)

    #if polygon1_lat and polygon1_lon:
    if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
        polygon_defined = True


    # Calculate statistics (only calculate CMZ stats if polygon is defined)
    if polygon_defined:
        cmz_onset_mean = calculate_cmz_averages(climatological_onset_doy, polygon1_lon, polygon1_lat)
    else:
        cmz_onset_mean = np.nan
    
    # Create edges for pcolormesh (cell boundaries)
    lon_edges = np.concatenate([lons - (lons[1]-lons[0])/2, [lons[-1] + (lons[1]-lons[0])/2]])
    lat_edges = np.concatenate([lats - (lats[1]-lats[0])/2, [lats[-1] + (lats[1]-lats[0])/2]])
    LON_edges, LAT_edges = np.meshgrid(lon_edges, lat_edges)
    
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
        


    # create figure obj
    fig, ax = plt.subplots(figsize=(8, 6))

    masked_onset = np.ma.masked_invalid(climatological_onset_doy.values)

    vmin, vmax = np.nanpercentile(masked_onset.compressed(), [5, 95])

    im = ax.pcolormesh(LON_edges, LAT_edges, masked_onset, 
                             cmap='OrRd', vmin=vmin, vmax=vmax, shading='flat')
    
    # Add India outline
    if shpfile_dir:
        india_boundaries = get_india_outline(shpfile_dir)
        for boundary in india_boundaries:
            india_lon, india_lat = boundary
            ax.plot(india_lon, india_lat, color='black', linewidth=map_lw)
    
    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)), 
                         fill=False, edgecolor='black', linewidth=polygon_lw)
        ax.add_patch(polygon)
    
    # Add text annotations for onset days
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            value = climatological_onset_doy.values[i, j]
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
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Day of Year')

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

