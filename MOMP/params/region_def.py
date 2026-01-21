import numpy as np


# define rectangular domain
def domain(region, **kwargs):
    # swap = False

    if region == "Ethiopia":
        lats = 3
        latn = 15
        lonw = 33
        lone = 48

    if region == "India":
        lats = 6.46
        latn = 35.51
        lonw = 68.11
        lone = 91.4

    return lats, latn, lonw, lone


def polygon_boundary(da):

    polygon1_lat, polygon1_lon = None, None

    orig_lat = da.lat.values
    orig_lon = da.lon.values

    lat_diff = abs(orig_lat[1]-orig_lat[0])

    if abs(lat_diff - 2.0) < 0.1:  # 2-degree resolution
        polygon1_lon = np.array([83, 75, 75, 71, 71, 77, 77, 79, 79, 83, 83, 89, 89, 85, 85, 83, 83])
        polygon1_lat = np.array([17, 17, 21, 21, 29, 29, 27, 27, 25, 25, 23, 23, 21, 21, 19, 19, 17])
        print("Using 2-degree CMZ polygon coordinates")

    elif abs(lat_diff - 4.0) < 0.1:  # 4-degree resolution
        polygon1_lon = np.array([86, 74, 74, 70, 70, 82, 82, 86, 86])
        polygon1_lat = np.array([18, 18, 22, 22, 30, 30, 26, 26, 18])
        print("Using 4-degree CMZ polygon coordinates")

    elif abs(lat_diff - 1.0) < 0.1:  # 1-degree resolution
        polygon1_lon = np.array([74, 85, 85, 86, 86, 87, 87, 88, 88, 88, 85, 85, 82, 82, 79, 79, 78, 78, 69, 69, 74, 74])
        polygon1_lat = np.array([18, 18, 19, 19, 20, 20, 21, 21, 21, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 21, 21, 18])
        print("Using 1-degree CMZ polygon coordinates")

    return polygon1_lat, polygon1_lon


def add_polygon(ax, da, polygon, return_polygon=False):
    from matplotlib.patches import Polygon
    from momp.params.region_def import polygon_boundary

    if polygon:
        polygon1_lat, polygon1_lon = polygon_boundary(da)

    if len(polygon1_lat) > 0 and len(polygon1_lon) > 0:
        polygon_defined = True

    # Add CMZ polygon only if defined
    if polygon_defined:
        polygon = Polygon(list(zip(polygon1_lon, polygon1_lat)),
                         fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(polygon)

    if return_polygon:
        return ax, polygon1_lat, polygon1_lon, polygon_defined
    else:
        return ax

