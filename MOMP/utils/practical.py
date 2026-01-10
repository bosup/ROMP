import importlib.resources
import inspect

def set_dir(folder):
    """
    set absolute directory path for a specific folder in MOMP
    """
    package = "MOMP"
    base_dir = importlib.resources.files(package)
    target_dir = (base_dir / folder).resolve()

    return target_dir



def restore_args(func, kwargs, bound_args):
    """
    Restore keyword-only parameters of `func` back into kwargs.
    """
    sig = inspect.signature(func)
    new_kwargs = dict(kwargs)

    for name, param in sig.parameters.items():
        if (
            param.kind is param.KEYWORD_ONLY
            and name in bound_args
            and name not in new_kwargs
        ):
            new_kwargs[name] = bound_args[name]

    return new_kwargs



