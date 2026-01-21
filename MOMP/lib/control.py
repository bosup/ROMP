import os
#import inspect
from dataclasses import fields
from typing import Union, Tuple, List
from pathlib import Path

from momp.lib.convention import Case, Setting
from momp.utils.printing import combi_to_str
#from momp.io.input import set_dir
from momp.utils.practical import set_dir


def init_dataclass(dc, dic):
    keys = dc.__annotations__.keys()
    subset = {key: dic[key] for key in keys if key in dic}
    return dc(**subset)


def modify_list_keys(dictionary):
    modified_keys = []
    for key, value in dictionary.items():
        if key.endswith("_list"):
            modified_key = key[:-5]  # Remove the last 5 characters "_list"
            modified_keys.append(modified_key)
    return modified_keys


def case_across_list(item, list1, list2):
    """find the corresponding item in list2 for a given item in list1"""
    if item in list1:
        index = list1.index(item)
        if index < len(list2):
            return list2[index]
    return None


def iter_list(dic, ext="_list"):
    layout_pool = []
    for field in dic["layout"]:
        lst = dic.get(field + ext)  # .copy()
        layout_pool.append(lst)
    return layout_pool


def years_tuple_clim(year_start: int, year_end: int) -> tuple[int, ...]:
    """
    Create a tuple of integers from year_start to year_end inclusive.
    """
    return tuple(range(year_start, year_end + 1))


def years_tuple_model(start_date: tuple[int,int,int], end_date: tuple[int,int,int]) -> tuple[int, ...]:
    """
    Create a tuple of years from start_date to end_date inclusive.
    Each date is a tuple like (year, month, day)
    """
    start_year = start_date[0]
    end_year = end_date[0]
    return tuple(range(start_year, end_year + 1))


def take_ensemble_members(
    members: Union[List[int], str]
) -> List[int]:
    """
    Normalize members into a list of ints.

    Accepts:
    - list of ints → returned as-is
    - string 'start-end' → expanded to list
    """
    # Case 1: already a list of ints
    if isinstance(members, list):
        return list(members)  # return a copy

    # Case 2: string range "1-5"
    if isinstance(members, str) and "-" in members:
        start, end = map(int, members.split("-"))
        return list(range(start, end + 1))

    raise TypeError("members must be list[int] or 'start-end' string")


#def restore_args(func, kwargs, bound_args):
#    """
#    Restore keyword-only parameters of `func` back into kwargs.
#    """
#    sig = inspect.signature(func)
#    new_kwargs = dict(kwargs)
#
#    for name, param in sig.parameters.items():
#        if (
#            param.kind is param.KEYWORD_ONLY
#            and name in bound_args
#            and name not in new_kwargs
#        ):
#            new_kwargs[name] = bound_args[name]
#
#    return new_kwargs


def make_case(dataclass, combi, dic):

    layout = dic["layout"]

    layout_dict = {key: None for key in layout}
    layout_dict.update(zip(layout_dict.keys(), combi))

    case_keys = [
        field.name for field in fields(dataclass)
    ]  # get defined keys in dataclass

    dic_list_keys = modify_list_keys(dic)  # return keys endwith _list, removing _list
    dic_list_keys = [
        item for item in dic_list_keys if item not in list(layout_dict.keys())
    ]
    list_keys = list(set(case_keys).intersection(set(dic_list_keys)))
    list_keys.remove("tolerance_days") # tolerance_days is paired with verification_window


    case = init_dataclass(dataclass, dic)
    case.update(layout_dict)

    value_list = []

    for key in list_keys:
        value = case_across_list(case.model, dic["model_list"], dic[key + "_list"])

        if key == "model_dir":
            if not Path(value).is_absolute():
                value = set_dir(value)

        value_list.append(value)

        #if key == "model_dir":
        #    case.fn_var = value

    #case_name = "{}".format("_".join(combi_to_str(combi)))
    case_name = combi_to_str(combi)
    case.case_name = case_name.replace(" ", "_")

    #if not dic['years']:
    if dic['years'] == 'All':
        case.years = years_tuple_model(dic['start_date'], dic['end_date'])

    #if not dic['years_clim']:
    if dic['years_clim'] == 'All':
        case.years_clim = years_tuple_clim(dic['start_year_clim'], dic['end_year_clim'])

    case.members = take_ensemble_members(dic['members'])

    value_dic = dict(zip(list_keys, value_list))
    case.update(value_dic)

    case.tolerance_days = case_across_list(case.verification_window, 
                                           dic["verification_window_list"], dic["tolerance_days_list"])

    return case



