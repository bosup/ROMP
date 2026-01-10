import importlib.resources
import os
from pathlib import Path

#from MOMP.io.input import set_dir
from MOMP.utils.practical import set_dir
from MOMP.lib.control import init_dataclass
from MOMP.lib.convention import Setting

from .parser import create_parser
#from .parser import find_param_file


package = "MOMP"
base_dir = importlib.resources.files(package)
print(f"package base dir {base_dir}")

config_file = set_dir("params/config.in")

if os.path.exists(config_file):
    print("config_file:", config_file)
else:
    print("config_file not found:", config_file)
    config_file = os.path.join(base_dir, "params/config.in")
    print("config_file:", config_file)
    if os.path.exists(config_file):
        print("config_file found:", config_file)
    else:
        print("config_file not found:", config_file)


with open(config_file, "r") as f:
    params_in = f.read()

params_in = "\n".join(
    line for line in params_in.splitlines() if not line.strip().startswith("#")
)

exec(params_in)

excluded_vars = {"f", "config_file_path", "params_in"}


#if not Path(globals()["dir_in"]).is_absolute():
#    dir_in = set_dir(globals()["dir_in"])
#
if not Path(globals()["ref_model_dir"]).is_absolute():
    dir_out = set_dir(globals()["ref_model_dir"])

if not Path(globals()["dir_out"]).is_absolute():
    dir_out = set_dir(globals()["dir_out"])

if not Path(globals()["dir_fig"]).is_absolute():
    dir_fig = set_dir(globals()["dir_fig"])

if not Path(globals()["obs_dir"]).is_absolute():
    obs_dir = set_dir(globals()["obs_dir"])

if globals().get("thresh_file") is not None:
    if not Path(globals()["thresh_file"]).is_absolute():
        thresh_file = set_dir(globals()["thresh_file"])

if globals().get("shpfile_dir") is not None:
    if not Path(globals()["shpfile_dir"]).is_absolute():
        shpfile_dir = set_dir(globals()["shpfile_dir"])


# print("dir_in = ",dir_in)

# dic = {var: getattr(config, var) for var in dir(config) if not var.startswith("__")}
# dic = {var: getattr(params_in, var) for var in dir(params_in) if not var.startswith("__")}
# dic = {var: globals()[var] for var in globals() if not var.startswith("__")}

dic = {
    var: globals()[var]
    for var in globals()
    if not var.startswith("__")  # Exclude special Python variables
    and not callable(globals()[var])  # Exclude functions or callable objects
    # and not isinstance(globals()[var], type(os))
    # and isinstance(globals()[var], (str, int, float, bool, Path))
    and var not in globals().get("__builtins__", {})
    and var not in excluded_vars and var != "excluded_vars"
}

# print("\ndic= ", dic)
# print(json.dumps(dic, indent=4))


#parser = create_parser(dic)
#args = parser.parse_args()

args = create_parser(dic)

#this block is to test -p param/param_user.py, need #from .parser import find_param_file
#param_path = find_param_file(args.param)
#print("\n param_path = ", param_path)
#spec = importlib.util.spec_from_file_location("param_module", str(param_path))
#param_module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(param_module)
#print("\n", param_module.json_structure)
#print("\n", param_module.model_list)

# 1. Start with the configuration from the file
cfg = dic.copy()

# 2. Get all parsed arguments as a dictionary
args_dict = vars(args)

# 3. Create a dictionary containing ONLY the key/value pairs that exist
#    in BOTH the parsed arguments AND the original config file (The Controlled Merge)
overrides = {
    key: value
    for key, value in args_dict.items()
    if key in cfg # Only update keys that were originally in the config
}

# 4. Apply the overrides
cfg.update(overrides)


setting = init_dataclass(Setting, cfg)

