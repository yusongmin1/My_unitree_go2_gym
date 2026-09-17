import yaml
import inspect
from dataclasses import dataclass

def to_dict(obj):
    if hasattr(obj, '__dict__'):
        rst = {}
        for k, v in obj.__dict__.items():
            rst[k] = to_dict(v)
        return rst
    elif hasattr(obj, '__iter__'):
        return [to_dict(v) for v in obj]
    else:
        return obj

def to_yaml(obj):
    return yaml.dump(to_dict(obj))

def _combined_function(f1, f2):
    def wrapper(*args, **kwargs):
        f1(*args, **kwargs)
        f2(*args, **kwargs)
    return wrapper
            
def check_type(obj):
    pass
            
def init_member_classes(obj):
    for key in dir(obj):
        if key=="__class__":
            continue
        var =  getattr(obj, key)
        if inspect.isclass(var):
            i_var = var()
            setattr(obj, key, i_var)
            init_member_classes(i_var)
                
def custom_init(obj):
    check_type(obj)
    # init_member_classes(obj)

def print_config(obj):
    key_colors = ["\033[94m", "\033[95m", "\033[96m", "\033[97m"]
    value_color = "\033[92m"
    border_color = "\033[91m"
    reset_color = "\033[0m"

    def format_key_value(key, value, prefix="", level=0):
        key_color = key_colors[level % len(key_colors)]
        if hasattr(value, "__dict__"):
            formatted = []
            for sub_key in dir(value):
                if sub_key.startswith("_"):
                    continue
                sub_value = getattr(value, sub_key)
                formatted.extend(format_key_value(sub_key, sub_value, f"{prefix}{key}.", level + 1))
            return formatted
        else:
            return [f"{key_color}{prefix}{key}{reset_color}: {value_color}{value}{reset_color}"]

    formatted_entries = []
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        formatted_entries.extend(format_key_value(key, value))

    max_length = max(60, max(len(entry) for entry in formatted_entries))
    border = f"{border_color}{'-' * (max_length + 4)}{reset_color}"
    print(border)
    for i in range(0, len(formatted_entries), 1):
        row = formatted_entries[i:i+1]
        row_str = " | ".join(row)
        if len(row_str) > max_length:
            row_str = "\n| ".join(row)
        print(f"{border_color}| {row_str.ljust(max_length)} |{reset_color}")
        if i + 1 < len(formatted_entries):
            print(border)
    print(border)

def config(cls, **kwargs):
    # setattr(cls, "__init__", _combined_function(cls.__init__, custom_init)) # FIXME has bug
    setattr(cls, 'to_dict', to_dict)
    setattr(cls, 'to_yaml', to_yaml)
    setattr(cls, 'print_config', print_config)
    cls = dataclass(cls, **kwargs)
    return cls

class _MISSING_TYPE:
    pass

MISSING = _MISSING_TYPE()