from dataclasses import asdict
from itertools import product

from momp.metrics.skill import create_score_results
from momp.graphics.heatmap import create_heatmap
from momp.graphics.reliability import plot_reliability_diagram
from momp.graphics.panel_portrait_skill import panel_portrait_bss_auc
from momp.io.output import save_score_results
from momp.lib.control import iter_list, make_case
from momp.lib.convention import Case
#from momp.lib.loader import cfg, setting
from momp.lib.loader import get_cfg, get_setting
#from momp.io.output import set_nested


#def bin_skill_score(BSS, RPS, AUC, skill_score, ref_model, ref_model_dir,
#                         years, years_clim, model, model_forecast_dir, obs_dir, thres_file
#                         members, max_forecast_day, day_bins, date_filter_year,
#                         file_pattern, mok, save_csv_score, plot_heatmap, **kwargs):

cfg, setting = get_cfg(), get_setting()

def skill_score_in_bins(cfg=cfg, setting=setting):

    # only execute for ensemble forecasts
    if not cfg.get('probabilistic'):
        return

    result_overall = {}
    result_binned = {}

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
            #save_score_results(score_results, **case_cfg)
            binned_data, overall_scores = save_score_results(score_results, **case_cfg)

        result_binned[case.model] = binned_data
        result_overall[case.model] = overall_scores
        

        # heatmap plot
        if case_cfg['plot_heatmap']:
            create_heatmap(score_results, **case_cfg)

        # reliability plot
        if case_cfg['plot_reliability']:
            plot_reliability_diagram(score_results["forecast_obs_df"], **case_cfg)

#    print("\n score_results \n ", score_results['skill_results']['bin_fair_brier_skill_scores'])
#    print("\n binned_data \n", binned_data['Fair_Brier_Skill_Score'])


#    if 2 > 1:
#        import pickle
#        fout = os.path.join(cfg['dir_out'],"combi_binned_skill_scores_results.pkl")
#        with open(fout, "wb") as f:
#            pickle.dump(result_binned, f)
#
#        fout = os.path.join(cfg['dir_out'],"combi_overall_skill_scores_results.pkl")
#        with open(fout, "wb") as f:
#            pickle.dump(result_overall, f)


    # panel heatmap plot for binned BSS and AUC
    if case_cfg['plot_panel_heatmap_skill']:
        panel_portrait_bss_auc(result_binned, **case_cfg)

    # bar plot for BSS, RPSS, AUC in window
    if case_cfg['plot_bar_bss_rpss_auc']:
        panel_bar_bss_rpss_auc(result_overall, **case_cfg)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    skill_score_in_bins()
