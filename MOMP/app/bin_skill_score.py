from dataclasses import asdict
from itertools import product

from MOMP.metrics.skill import create_score_results
from MOMP.graphics.heatmap import create_heatmap
from MOMP.graphics.reliability import plot_reliability_diagram
from MOMP.io.output import save_score_results
from MOMP.lib.control import iter_list, make_case
from MOMP.lib.convention import Case
from MOMP.lib.loader import cfg, setting


#def bin_skill_score(BSS, RPS, AUC, skill_score, ref_model, ref_model_dir,
#                         years, years_clim, model, model_forecast_dir, obs_dir, thres_file
#                         members, max_forecast_day, day_bins, date_filter_year,
#                         file_pattern, mok, save_csv_score, plot_heatmap, **kwargs):


def skill_score_in_bins(cfg=cfg, setting=setting):

    # only execute for ensemble forecasts
    if not cfg.get('probabilistic'):
        return

    layout_pool = iter_list(cfg)

    for combi in product(*layout_pool):
        case = make_case(Case, combi, cfg)

        print(f"processing bin skill score for {case.case_name}")

        case_cfg = {**asdict(setting), **asdict(case)}

#        from pprint import pprint
#        pprint(case_cfg)

        # Create bin skill score metrics
        score_results = create_score_results(**case_cfg)
        
        # save score results as csv file
        if case_cfg['save_csv_score']:
            #save_score_results(score_results, model, **case_cfg)
            save_score_results(score_results, **case_cfg)
        
        # heatmap plot
        if case_cfg['plot_heatmap']:
            create_heatmap(score_results, **case_cfg)

        # reliability plot
        if case_cfg['plot_reliability']:
            plot_reliability_diagram(score_results["forecast_obs_df"], **case_cfg)


