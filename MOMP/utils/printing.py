from importlib.metadata import version as pkg_version

def combi_to_str(combi, sep="_", tuple_sep="-", suffix=""):
    """
    Convert a tuple like ('a', (1,15), 'X') into a string tag:
    - Strings or numbers → as is
    - Tuples → joined with `tuple_sep`
    """
    parts = []
    for item in combi:
        if isinstance(item, tuple):
            # join tuple elements with tuple_sep
            parts.append(tuple_sep.join(map(str, item)))
        else:
            parts.append(str(item))
    return sep.join(parts) + suffix


def tuple_to_str(item):
    return "-".join(map(str, item))


def tuple_to_str_range(item):
    """ ignore all middle elements and only join the first and last items of a tuple """
    if len(item) == 0:
        return ""
    elif len(item) == 1:
        return str(item[0])
    else:
        return f"{item[0]}-{item[-1]}"



def print_momp_banner(cfg):

    version = pkg_version("momp")
    project_name = cfg.get("project_name")

    banner = f"""
================================================================================
  __  __   ___   __  __   ____
 |  \/  | / _ \ |  \/  | |  _ \\
 | |\/| || | | || |\/| | | |_) |
 | |  | || |_| || |  | | |  __/
 |_|  |_| \___/ |_|  |_| |_|

 Monsoon Onset Metrics Package (MOMP)
 Version : {version}

--------------------------------------------------------------------------------
 Project    : {project_name}
 Start Time : {__import__('datetime').datetime.now().isoformat(timespec='seconds')}
--------------------------------------------------------------------------------

 Initializing analysis pipeline...
================================================================================
"""
    print(banner)

