import os
#import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from momp.lib.loader import get_cfg, get_setting
from momp.io.output import nested_dict_to_array, analyze_nested_dict
from momp.utils.visual import portrait_plot


def panel_portrait_mae_far_mr(results, *, dir_fig, **kwargs):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    
    # load mean mae data and calculate ananomly wrt cimatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "mean_mae")
    data = arr[:-1] - arr[-1]
    
    fig, ax1, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax1, 
                                 annotate=True, annotate_data=data, title='$\Delta MAE$ (days)', colorbar_off=True)
    
    # anomaly false alarm rate relative to climatology 
    arr, row_labels, col_labels = nested_dict_to_array(results, "false_alarm_rate")
    data = arr[:-1] - arr[-1]
    
    fig, ax2, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax2, 
                                 annotate=True, annotate_data=data, title='$\Delta FAR (\%)$', colorbar_off=True)#, cbar_kw={"orientation":"horizontal"})
    
    ax2.set_xlabel('Forecast window (days)')
    
    # anomaly miss rate relative to climatology
    arr, row_labels, col_labels = nested_dict_to_array(results, "miss_rate")
    data = arr[:-1] - arr[-1]
    
    fig, ax3, im = portrait_plot(data, col_labels, row_labels[:-1], fig=fig, ax=ax3, 
                                 annotate=True, annotate_data=data, title='$\Delta MR (\%)$', colorbar_off=True)
    
    fig.tight_layout()
    plt.show()

    # save figure
    figure_filename = f'panel_portrait_mae_far_mr.png'
    figure_filename = os.path.join(dir_fig, figure_filename)
    fig.savefig(figure_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{figure_filename}'")

    return fig, (ax1, ax2, ax3)



if __name__ == "__main__":

    from itertools import product
    from momp.lib.control import iter_list, make_case
    from momp.utils.printing import tuple_to_str
    import xarray as xr

    cfg, setting = get_cfg(), get_setting()
    
    result = {}
    
    layout_pool = iter_list(cfg)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, cfg)

        fi = os.path.join(cfg['dir_out'],"spatial_metrics_{}.nc")
        fi = fi.format(case.case_name)
        ds = xr.open_dataset(fi)
        #ds_subset = ds[["mean_mae", "false_alarm_rate", "miss_rate"]]
        var_list = ["mean_mae", "false_alarm_rate", "miss_rate"]

        day_bin = tuple_to_str(case.verification_window) 
        for var in var_list:
            result[case.model][day_bin][var] = ds[var]

        fi = os.path.join(cfg['dir_out'],"spatial_metrics_{}_{}.nc")
        fi = fi.format(case.ref_model, day_bin)
        if result.get(case.ref_model, {}).get(day_bin) is not None:
            ds = xr.open_dataset(fi)
            var_list = ["mean_mae", "false_alarm_rate", "miss_rate"]
            for var in var_list:
                result[case.ref_model][day_bin][var] = ds[var]

    
#    fout = os.path.join(cfg['dir_out'],"combi_error_results.pkl")
#    with open(fout, "rb") as f:
#        import pickle
#        results = pickle.load(f)
    
    panel_portrait_mae_far_mr(results, **cfg)


#mae = nested_dict_to_array(results, "mean_mae") # "miss_rate", "false_alarm_rate"
#print(mae)

#model_list = cfg["model_list"]
#window_list = cfg["verification_window_list"]
