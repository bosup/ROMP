import numpy as np


def extract_binned_dict(dic, second_level_key, bin_key="clean_bins"):
    """
    extract a 2-D array from a 3-level nested dictionary
    given a specific 2nd-level key, 
    Extract all 1st-level keys
    And get all 3rd-level values
    """
    #array_2d = [
    #        list(dic[first_key][second_level_key].values())
    #        for first_key in dic.keys()
    #]

    # 1st-level keys
    first_level_keys = list(dic.keys())
    
    #print(type(next(iter(dic.values()))[second_level_key]))
    #print(next(iter(dic.values()))[second_level_key])

    # 3rd-level keys (assume all 1st-level keys have the same 3rd-level keys)
    #third_level_keys = list(next(iter(dic.values()))[second_level_key].keys())

    # binned skill scores are saved as a list without bin keys
    # use "clean_bins" key to indicate bins
    third_level_keys = next(iter(dic.values())).get(bin_key)
    
    # Build 2-D array (rows = 1st-level, columns = 3rd-level)
    #array_2d = np.array([
    #    [dic[first][second_level_key][third] for third in third_level_keys]
    #    for first in first_level_keys
    #])

    array_2d = np.array([
        dic[first][second_level_key]
        for first in first_level_keys
    ])
    return array_2d, first_level_keys, third_level_keys



def extract_overall_dict(dic, second_level_key):
    """
    extract a 1-D array from a 2-level nested dictionary
    given a specific 2nd-level key, 
    Extract all 1st-level keys
    And get all 2nd-level values
    """
    # 1st-level keys
    first_level_keys = list(dic.keys())

#    print("\n first_level_keys = ", first_level_keys)
#    print("\n first key value dict = ", next(iter(dic.values())))
#    print("\n first key value dict = ", next(iter(dic.values())).get(second_level_key))
    
    # 1-D array of values for the chosen 2nd-level key
    #array_1d = np.array([dic[first][second_level_key][0] for first in first_level_keys])
    array_1d = np.array([dic[first][second_level_key] for first in first_level_keys])


    return array_1d, first_level_keys




