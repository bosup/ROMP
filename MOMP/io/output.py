#from pathlib import Path
import os
import numpy as np
import pandas as pd
import pickle
from MOMP.stats.bins import get_target_bins
#from MOMP.utils.printing import tuple_to_str

#def file_path(directory, filename):
#    """Join directory and filename into a full path."""
#    return Path(directory) / filename


def save_score_results(score_results, *, model, max_forecast_day, dir_out, **kwargs):
    """Save overall and binned skill scores to CSV files"""
            
    # Save overall scores
    overall_scores = {
        'Metric': ['AUC', 'Fair Brier Score', 'Fair Brier Skill Score', 'Fair RPS', 'Fair RPS Skill Score'],
        'Score': [
            score_results['AUC']['auc'],
            score_results['BS']['fair_brier_score'],
            score_results['skill_results']['fair_brier_skill_score'],
            score_results['RPS']['fair_rps'],
            score_results['skill_results']['fair_rps_skill_score']
        ]
    }

    overall_df = pd.DataFrame(overall_scores)
    overall_filename = f'overall_skill_scores_{model}_{max_forecast_day}day.csv'
                        #{model}_{tuple_to_str(verification_window)}window_{max_forecast_day}day.csv'
    overall_filename = os.path.join(dir_out, overall_filename)
    overall_df.to_csv(overall_filename, index=False)
    print(f"Saved overall scores to '{overall_filename}'")
    print(overall_df)

    # Get target bins
    target_bins = get_target_bins(score_results['BS'], score_results['BS_ref'])

    # Save binned scores
    binned_data = {
        'Bin': target_bins,
        'Fair_Brier_Skill_Score': [score_results['skill_results']['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'AUC': [score_results['AUC']['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Forecast': [score_results['BS']['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Climatology': [score_results['BS_ref']['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    }

    binned_df = pd.DataFrame(binned_data)
    binned_filename = f'binned_skill_scores_{model}_{max_forecast_day}day.csv'
    binned_filename = os.path.join(dir_out, binned_filename)
    binned_df.to_csv(binned_filename, index=False)
    print(f"Saved binned scores to '{binned_filename}'")
    print(binned_df)



def save_ref_score_results(results, filename):
                           #*, model, verification_window, max_forecast_day, dir_out, **kwargs):

    to_save = {
    "climatology_obs_df": results["climatology_obs_df"],
    "brier_ref": results["BS_ref"],
    "rps_ref": results["RPS_ref"],
    "auc_ref": results["AUC_ref"],
    }
    
    #filename = f'ref_scores_{model}_{tuple_to_str(verification_window)}window_{max_forecast_day}day.csv'
    #filename = os.path.join(dir_out, filename)

    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(to_save, f)
    


def load_ref_score_results(ref_score_file, results_dict):

#if ref_score_file.exists():
    with ref_score_file.open("rb") as f:
        loaded_results = pickle.load(f)

    results_dict.update(loaded_results)

    return results_dict
    
