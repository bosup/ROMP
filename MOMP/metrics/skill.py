#from momp.io.output import save_score_results
from momp.stats.climatology import compute_climatological_onset_dataset
from momp.stats.bins import multi_year_forecast_obs_pairs, multi_year_climatological_forecast_obs_pairs
from momp.stats.score import calculate_brier_score, calculate_auc, calculate_rps, calculate_brier_score_climatology, calculate_auc_climatology, calculate_skill_scores
from momp.utils.printing import tuple_to_str
from momp.io.output import save_ref_score_results, load_ref_score_results
#from momp.lib.control import restore_args
from momp.utils.practical import restore_args

def create_score_results(*, BS, RPS, AUC, skill_score, 
                         ref_model, ref_model_dir, ref_model_var, ref_model_file_pattern, ref_model_unit_cvt,
                         years, years_clim, obs_dir, obs_file_pattern, obs_var,
                         thresh_file, thresh_var, wet_threshold,
                         date_filter_year, init_days, start_date, end_date,
                         model_dir, model_var, unit_cvt, file_pattern,
                         wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, fallback_date, mok,
                         members, onset_percentage_threshold, max_forecast_day, day_bins, **kwargs):

#    print("="*60)
#    print("S2S MONSOON ONSET SKILL SCORE ANALYSIS")
#    print("="*60)
#    print(f"Model: {model}")
#    print(f"Years: {years}")
#    print(f"Max forecast day: {max_forecast_day}")
#    print(f"Day bins: {day_bins}")
#    print(f"MOK filter: {mok}")
#    print("="*60)

    kwargs = restore_args(create_score_results, kwargs, locals())
#    from pprint import pprint
#    print(kwargs)

    results = {}

    print("\n1. Processing forecast model...")
    forecast_obs_df = multi_year_forecast_obs_pairs(**kwargs)
    results["forecast_obs_df"] = forecast_obs_df

    results["BS"] = calculate_brier_score(forecast_obs_df) if BS else None
    results["RPS"] = calculate_rps(forecast_obs_df) if RPS else None
    results["AUC"] = calculate_auc(forecast_obs_df) if AUC else None


    print("\n2. Processing reference model...")

    results["BS_ref"], results["RPS_ref"], results["AUC_ref"] = None, None, None
    results['skill_score'] = None

    # check if ref_results exist
    #dir_out = kwargs.get("dir_out")
    #model = kwargs.get("model")
    ##verification_window = kwargs.get("verification_window")
    ##filename = f'ref_scores_{model}_{tuple_to_str(verification_window)}window_{max_forecast_day}day.csv'
    #filename = f'ref_scores_{model}_{max_forecast_day}day.csv'
    #filename = os.path.join(dir_out, f"{filename}.pkl")

    #if filename.exists():
    #    results = load_ref_score_results(filename, results)
    #    if skill_score:
    #        skill_results = calculate_skill_scores(
    #        results["BS"], results["RPS"],
    #        results["BS_ref"], results["RPS_ref"]
    #        )
    #        results["skill_results"] = skill_results
    #    return results

    if ref_model == "climatology":

        clim_onset = compute_climatological_onset_dataset(**kwargs)
        climatology_obs_df = multi_year_climatological_forecast_obs_pairs(clim_onset, **kwargs)
        results["climatology_obs_df"] = climatology_obs_df
        
        if BS:
            brier_ref = calculate_brier_score_climatology(climatology_obs_df)
            results["BS_ref"] = brier_ref
        else:
            results["BS_ref"] = None
        
        if RPS:
            rps_ref = calculate_rps(climatology_obs_df)
            results["RPS_ref"] = rps_ref
        else:
            results["RPS_ref"] = None

        if AUC:
            auc_ref = calculate_auc_climatology(climatology_obs_df)
            results["AUC_ref"] = auc_ref
        else:
            results["AUC_ref"] = None

    else:
        results["climatology_obs_df"] = None

        kwargs_ref = {**kwargs,
                      'model': ref_model,
                      'model_dir': ref_model_dir,
                      'model_var': ref_model_var,
                      'file_pattern': ref_model_file_pattern,
                      'unit_cvt': ref_model_unit_cvt
                      }

        ref_obs_df = multi_year_forecast_obs_pairs(**kwargs_ref)

        if BS:
            brier_ref = calculate_brier_score(ref_obs_df)
            results["BS_ref"] = brier_ref
        else:
            results["BS_ref"] = None
        
        if RPS:
            rps_ref = calculate_rps(ref_obs_df)
            results["RPS_ref"] = rps_ref
        else:
            results["RPS_ref"] = None

        if AUC:
            auc_ref = calculate_auc(ref_obs_df)
            results["AUC_ref"] = auc_ref
        else:
            results["AUC_ref"] = None


    #save_ref_score_results(results, filename)


    print("results[BS]  =  ", results["BS"])
    print("results[BS_ref]  =  ", results["BS_ref"])
    print("results[RPS]  =  ", results["RPS"])
    print("results[RPS_ref]  =  ", results["RPS_ref"])
    if skill_score:

        skill_results = calculate_skill_scores(
        results["BS"], results["RPS"],
        results["BS_ref"], results["RPS_ref"]
        )

        results["skill_results"] = skill_results

#    if save_csv_score:
#        save_score_results(results, model, max_forecast_day))


    return results

