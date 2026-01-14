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
from MOMP.graphics.onset_map import plot_spatial_climatology_onset
from MOMP.io.output import save_metrics_to_netcdf
#from MOMP.io.output import file_path
from MOMP.io.output import set_nested
from MOMP.utils.printing import tuple_to_str_range


def spatial_far_mr_mae_map(cfg=cfg, setting=setting):#, **kwargs):

    # only executed for deterministic forecasts
    if cfg.get('probabilistic'):
        return

    layout_pool = iter_list(cfg)

    results = {}

    for combi in product(*layout_pool):
        case = make_case(Case, combi, cfg)

        print(f"processing model onset evaluation for {case.case_name}")

        case_cfg = {**asdict(case), **asdict(setting)}

        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg)
        print("AAAAA")
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        #current = results
        #for key in combi[:-1]:
        #    current = current.setdefault(key, {})
        #current[combi[-1]] = value


        # Save spatial metrics to NetCDF
        if case_cfg["save_nc_spatial_far_mr_mae"]:
        
            desc_dict = {
                    'title': 'Monsoon Onset MAE, FAR, MR Analysis',
                    'description': """Spatial maps of Mean Absolute Error, False Alarm Rate, and Miss Rate 
                    for monsoon onset predictions""",
            }

            save_metrics_to_netcdf(spatial_metrics, case_cfg, desc_dict=desc_dict)


        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg)


#        # make climatological onset plot
#        #if case.model=='climatology' and case_cfg['plot_climatology_onset']:
#        if case_cfg['plot_climatology_onset']:
#            plot_spatial_climatology_onset(onset_da_dict, **case_cfg)


    # ------------------------------------------------------------------------
    # baseline metrics (climatology or user specified model)

    if not cfg['ref_model']:
        return 

    cfg_ref = cfg
    cfg_ref['model_list'] = (cfg['ref_model'],)
    layout_pool = iter_list(cfg_ref)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, cfg_ref)
        print(f"processing model onset evaluation for {case.case_name}")
        case_cfg = {**asdict(case), **asdict(setting)}

        case_cfg_ref = {**case_cfg,
                      #'model': case_cfg['ref_model'],
                      'model_dir': case_cfg['ref_model_dir'],
                      'model_var': case_cfg['ref_model_var'],
                      'file_pattern': case_cfg['ref_model_file_pattern'],
                      'unit_cvt': case_cfg['ref_model_unit_cvt']
                      }

        # model-obs onset benchmarking
        metrics_df_dict, onset_da_dict = compute_metrics_multiple_years(**case_cfg_ref)
        
        # Create spatial metrics
        spatial_metrics = create_spatial_far_mr_mae(metrics_df_dict, onset_da_dict)
        
        # log case result into combined multi-case results dictionary
        results = set_nested(results, combi, spatial_metrics)

        # make spatial metrics plot
        if case_cfg['plot_spatial_far_mr_mae']:
            plot_spatial_metrics(spatial_metrics, **case_cfg_ref)


    # save climatological onset to netcdf
    if case_cfg_ref['save_nc_climatology']:
        fout = os.path.join(case_cfg_ref['dir_out'], "climatology_onset_doy_{}.nc")
        fout = fout.format(tuple_to_str_range(case_cfg_ref['years_clim']))
        climatological_onset_doy = next(iter(onset_da_dict.values()))
        climatological_onset_doy.to_netcdf(fout)

    # spatial map of climatology onset day
    if case_cfg['plot_climatology_onset']:
        plot_spatial_climatology_onset(onset_da_dict, **case_cfg_ref)

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    spatial_far_mr_mae_map()
