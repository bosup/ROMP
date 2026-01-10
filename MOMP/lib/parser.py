import argparse
#from pathlib import Path

def create_parser(config):
    parser = argparse.ArgumentParser()

    #parser.add_argument("-p", "--param", required=True)

    parser.add_argument(
        "-p",
        "--param",
        type=str,
        default="params/config.in",
        help="Parameter file path (default: params/config.in)"
    )

    parser.add_argument(
        "--model_list",
        nargs="+",
        default=config["model_list"],
        help=f"Model list (default: {config['model_list']})"
    )

#    parser.add_argument(
#        "--ensemble_list",
#        nargs="+",
#        type=int,
#        default=config["ensemble_list"],
#        help=f"Ensemble list (default: {config['ensemble_list']})"
#    )
#
#    parser.add_argument(
#        "--season_start",
#        type=str,
#        default=config["season_start"],
#        help=f"Season start (default: {config['season_start']})"
#    )
#
#    # Boolean argument with dynamic default
#    parser.add_argument(
#        "--MAE",
#        type=bool,
#        default=config["MAE"],
#        help=f"Use MAE (default: {config['MAE']})"
#    )


    args = parser.parse_args()

    # ---- Convert list → tuple if needed ----
    # Only convert if user supplied; default already tuple
    if isinstance(args.model_list, list):
        args.model_list = tuple(args.model_list)

#    if isinstance(args.ensemble_list, list):
#        args.ensemble_list = tuple(args.ensemble_list)

#    return parser
    return args


#reserved for python -p option in loader.py need from pathlib import Path 
#def find_param_file(filename):
#    # First, check cwd
#    f = Path(filename)
#    if f.is_file():
#        return f.resolve()
#
#    # Then check package subdirectory 'params' relative to this script
#    package_dir = Path(__file__).parent
#    f2 = package_dir / "params" / filename
#    if f2.is_file():
#        return f2.resolve()
#
#    print(f"Parameter file {filename} not found")
#    sys.exit(1)



