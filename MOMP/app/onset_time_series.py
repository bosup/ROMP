import os
from momp.io.input import load_imd_rainfall, load_thresh_file
from momp.stats.detect import detect_observed_onset
#from momp.lib.loader import cfg,setting
from momp.lib.loader import get_cfg, get_setting
from momp.graphics.rainfall_time_series import plot_rainfall_timeseries_with_onset_and_wetspell
from momp.utils.practical import restore_args


cfg, setting = get_cfg(), get_setting()

def obs_onset_analysis(year, **kwargs):

    kwargs = restore_args(obs_onset_analysis, kwargs, locals())

    # load onset precip threshold, 2-D or scalar
    thresh_da = load_thresh_file(**kwargs)

    #year = years[0]
    print(f"\n{'='*50}")
    print(f"Processing year {year}")
    print(f"{'='*50}")

    # obs onset
    da = load_imd_rainfall(year, **kwargs)
    onset_da = detect_observed_onset(da, thresh_da, year, **kwargs)

    start_date = kwargs["start_date"][1:]
    end_date = kwargs["end_date"][1:]

    da_sub = da.where(
        (
            (da.time.dt.month > start_date[0]) |
            ((da.time.dt.month == start_date[0]) & (da.time.dt.day >= start_date[1]))
        ) &
        (
            (da.time.dt.month < end_date[0]) |
            ((da.time.dt.month == end_date[0]) & (da.time.dt.day <= end_date[1]))
        ),
        drop=True
    )


#    imd = da
#    print(imd)
#    print(onset_da)

#    fully_valid_mask = imd.notnull().all(dim="time")
#    
#    fully_valid_locations = (
#        fully_valid_mask
#        .where(fully_valid_mask, drop=True)
#        .stack(points=("lat", "lon"))
#        .to_dataframe(name="valid")
#        .reset_index()[["lat", "lon"]]
#    )
#    
#    print(fully_valid_locations)


    save_path = os.path.join(kwargs["dir_fig"], "onset_time_series.png") 

    #plot_onset_time_series(lat=10, lon=40,)
    plot_rainfall_timeseries_with_onset_and_wetspell(da_sub, onset_da, None, 
                                                     #lat_select=32, lon_select=72, year_select=year)
                                                     lat_select=20, lon_select=80, year_select=year, save_path=save_path)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    obs_onset_analysis(year=2013, **cfg)

