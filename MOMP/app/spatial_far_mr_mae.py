import os
import xarray as xr
from dataclasses import asdict
from itertools import product

from MOMP.metrics.error import create_spatial_far_mr_mae
from MOMP.stats.benchmark import compute_metrics_multiple_years
from MOMP.lib.control import iter_list, make_case
from MOMP.lib.convention import Case
from MOMP.lib.loader import cfg,setting
from MOMP.graphics.maps import plot_spatial_metrics
#from MOMP.io.output import file_path



def spatial_far_mr_mae_map(cfg=cfg, setting=setting):#, **kwargs):

    # only executed for deterministic forecasts
    if cfg.get('probabilistic'):
        return

    layout_pool = iter_list(cfg)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, cfg)

        print(f"processing model onset evaluation for {case.case_name}")

        case_cfg = {**asdict(case), **asdict(setting)}


        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg)
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        if case_cfg["save_nc_spatial_far_mr_mae"]:
            #fout = file_path(kwargs['dir_out'], kwargs['fout_spatial_far_mr_mae'])
            #fout = os.path.join(setting.dir_out, setting.fout_spatial_far_mr_mae)
            fout = os.path.join(case_cfg['dir_out'], "spatial_metrics_{}.nc")
            fout = fout.format(case.case_name)
        
            ds = xr.Dataset(spatial_metrics)

            # Add global attributes
            ds.attrs['title'] = 'Monsoon Onset MAE, FAR, MR Analysis'
            ds.attrs['description'] = ('Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate '
                                        'for monsoon onset predictions'
                                       )
            ds.attrs['model'] = case.model
            ds.attrs['years'] = str(case_cfg['years'])
            ds.attrs['tolerance_days'] = case_cfg['tolerance_days']
            ds.attrs['verification_window'] = case_cfg['verification_window']
            ds.attrs['max_forecast_day'] = case_cfg['max_forecast_day']
            ds.attrs['mok_filter'] = int(bool((case_cfg['mok'])))  # Convert boolean to integer (0 or 1)
        
            # Save to NetCDF
            ds.to_netcdf(fout)
            print(f"\nSpatial metrics saved to: {fout}")


        if case_cfg['plot_spatial_far_mr_mae']:

            plot_spatial_metrics(spatial_metrics, **case_cfg)




