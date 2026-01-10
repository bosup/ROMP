import xcdat as xc
import xarray as xr
import numpy as np
from MOMP.params.region_def import domain

#def domain(region, **kwargs):
#    # swap = False
#
#    if region == "Ethiopia":
#        lats = 3
#        latn = 15
#        lonw = 33
#        lone = 48
#
#    return lats, latn, lonw, lone


def lon_swap(ds_tag, *, region, **kwargs):
    lats, latn, lonw, lone = domain(region, **kwargs)

    coord_list = list(ds_tag.coords.keys())

    if "lon" not in coord_list:
        print("lon NOT in coords for model --> ")  # , model_name)
        lat_coords = [variable for variable in coord_list if "lat" in variable][0]
        lon_coords = [variable for variable in coord_list if "lon" in variable][0]

        ds_tag = ds_tag.rename({lat_coords: "lat", lon_coords: "lon"})

    if np.min(ds_tag.lon) < 0:
        # print('swap lonw, lone to -180,180')
        # print('lonw = ',lonw,' lone = ',lone)
        lonw = ((lonw + 180) % 360) - 180
        lone = ((lone + 180) % 360) - 180
        # print('conformed lonw = ',lonw,' lone = ',lone)

    # swap = False

    if lonw > lone:
        # swap = True
        # print('lonw > lone, then swap')

        if np.min(ds_tag.lon) < 0:
            ds_tag = xc.swap_lon_axis(ds_tag, (0, 360)).compute()
            lonw = lonw % 360
            lone = lone % 360
            # print('swapped lonw = ',lonw,' lone = ',lone)
        else:
            ds_tag = xc.swap_lon_axis(ds_tag, (-180, 180)).compute()
            lonw = ((lonw + 180) % 360) - 180
            lone = ((lone + 180) % 360) - 180
            # print('swapped lonw = ',lonw,' lone = ',lone)

        print("swapped longitude range ", np.min(ds_tag.lon), " - ", np.max(ds_tag.lon))

    return lats, latn, lonw, lone, ds_tag


def lat_swap(ds_tag):
    if ds_tag.lat[0] > ds_tag.lat[-1]:
        # swap = True
        ds_tag = ds_tag.isel(lat=slice(None, None, -1))

    return ds_tag


def coords_fmt(ds_tag, *, region, **kwargs):
    #ds_tag = time_swap(ds_tag)
    lats, latn, lonw, lone, ds_tag = lon_swap(ds_tag, region=region, **kwargs)
    ds_tag = lat_swap(ds_tag)

    return lats, latn, lonw, lone, ds_tag


def region_select(ds, *, region, **kwargs):

    lats, latn, lonw, lone, ds = coords_fmt(ds, region=region)

    ds_reg = ds.sel(lat=slice(lats, latn), lon=slice(lonw, lone))

    return ds_reg



