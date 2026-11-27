
import os
import json
import re

import traceback
from bson.binary import Binary
import pickle

try:
    import cloudpickle
except:
    print("Failed to import cloudpickle. Regular pickle should still work")
import dill
try:
    import yaml
except:
    pass
try:
    import tomli
except:
    pass
import pathlib
import platform
import gzip
import pandas as pd

import numpy as np
import hashlib
import inspect


import zipfile
from copy import deepcopy
from _mri_toolbox.framework.vault import Vault
import base64

from _mri_toolbox.utils.system import print

from collections import UserDict

class CaseInsensitiveDict(UserDict):
    def __init__(self, data=None, **kwargs):
        super().__init__()
        if data:
            self.update(data)
        if kwargs:
            self.update(kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __delitem__(self, key):
        return super().__delitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())

    def get(self, key, default=None):
        return super().get(key.lower(), default)

    def update(self, other=None, **kwargs):
        if other:
            if isinstance(other, dict):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwargs.items():
            self[key] = value

def camel_to_snake(name):
    """
    Convert label from camelCase to snake_case
    :param name:
    :return:
    """
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(name):
    """
    Convert from snake_case to camelCase
    :param name:
    :return:
    """
    parts = name.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

def to_camel(name):
    """
    Convert to true camelCase
    :param name:
    :return:
    """
    return snake_to_camel(camel_to_snake(name))


def lower_first(name):
    """
    Make the first letter lower
    :param name:
    :return:
    """
    return f"{name[0].lower()}{name[1:]}"


def normalize_keys(data, norm=camel_to_snake):
    """
    Normalize the keys of a dict given norm
    :param data:
    :param norm:
    :return:
    """
    new_data = {}
    if isinstance(norm, str):
        if norm == 'camel_to_snake':
            norm = camel_to_snake
        elif norm == 'to_camel':
            norm = to_camel
        elif norm == 'lower_first':
            norm = lower_first
    if isinstance(data, dict):
        new_data = {norm(k): v for k, v in data.items()}

    return new_data


class EncodingException(Exception):
    pass

def encode_object(obj, callable_encoder='pickle'):
    if isinstance(obj, Vault):
        return "Vault content is protected"
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    try:
        if obj.__class__.__name__ == 'MultiValue':
            return list(obj)
    except:
        pass
    if callable(obj):
        try:
            if callable_encoder == 'pickle':
                return str(pickle.dumps(obj))
            elif callable_encoder == 'dill':
                return str(dill.dumps(obj))
        except:
            return inspect.getsource(obj)
    raise EncodingException

class CustomEncoderWithDill(json.JSONEncoder):
    def default(self, obj):
        try:
            return encode_object(obj, 'dill')
        except EncodingException as e:
            return json.JSONEncoder.default(self, obj)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return encode_object(obj, 'pickle')
        except EncodingException as e:
            return json.JSONEncoder.default(self, obj)


def compile_function_from_string(fct):
    if not isinstance(fct, str):
        return fct
    #
    # check if it is a callable python type which should not be callable
    if fct in ['float', 'int', 'complex']:
        return fct
    #
    # check if the fct is a stringified function or lambda
    if 'def ' in fct:
        if not fct.startswith('def '):
            print('[!] Custom function is not correctly defined. Should start with \"def \"')
        fct_name = fct.split('(')[0].split()[-1]
        exec(fct)
        return locals()[fct_name]
    elif 'lambda ' in fct:
        operation = eval(fct)
        return operation
    else:
        try:
            return eval(fct)
        except:
            return fct


def _dump_manage_ext_gzip(path, ext, verbose=True):
    """
    Format the path of gzip compressed files, for any type of extension.
    :param path [str]: save path.
    :param ext [str]: type of the non-compressed file (json, pickle, dill, ...).
    :param verbose [bool]: verbosity.
    :return:
    """
    # Compressed extension
    cext = 'gzip'
    #
    path_split = path.split('.')
    # Replace gz with gzip
    path_split = ['gzip' if x == 'gz' else x for x in path_split]
    #
    if   len(path_split)==1:
        path_out = f'{path}.{ext}.{cext}'
    elif path_split[-2] == ext and path_split[-1] == cext:
        path_out = '.'.join(path_split)
    elif path_split[-1] == ext and path_split[-2] == cext:
        path_out = '.'.join(path_split[:-2] + [ext, cext])
    elif path_split[-1] == ext:
        path_out = f'{path}.{cext}'
    elif path_split[-1] == cext:
        path_out = '.'.join(path_split[:-1] + [ext, cext])
    else:
        path_out = f'{path}.{ext}.{cext}'
    #
    if verbose:
        if path_out != path:
            print(f'The compressed {ext} filename has been set to: {path_out}')
    return path_out


def str2obj(str_obj, keep_going=False):
    '''
    Convert json-like-string to object. Single-quotes are supported allowing to avoid escaping double quotes

    :param str:
    :return:
    '''
    if not keep_going:
        if not str_obj is None:
            str_obj = str_obj.replace('\'', '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
            return json.loads(str_obj)
        else:
            return {}
    #
    try:
        if not str_obj is None:
            str_obj = str_obj.replace('\'', '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
            return json.loads(str_obj)
    except Exception as ex:
        traceback.print_exc()
        print(f'[str2obj] failed parsing {str_obj}: {ex}')
    return {}


def update_nested_dict(d, u, extend_list=False, inplace_replace_list=False, copy=False):
    """
    Recursively override u into base dict d.

    Parameters:
    - d (dict, list): target dictionary to update in place.
    - u (dict, list, variable): override dictionary whose values take precedence.
    - extend_list (bool): if True, append extra items from u when merging lists.
    - copy (bool): if True, copy the value to not overwrite the initial dictionary. Eq. ~inplace.

    Merge rules:
    1. Dict vs Dict: recurse into keys.
    1.5 List vs dict: dict will update each element of the list if it is a dict, otherwise the list will be extended.
    2. List vs List: merge element-wise up to min lengths.
       • If extend_list=True, append remaining u items.
       • Elif inplace_replace_list=True, ignore extra u items beyond d’s length.
       • Else, we just replace values.
    3. Otherwise: u’s value replaces d’s.
    """

    inplace = lambda x: x if not copy else deepcopy(x)  # inplace modification or deep copy to keep original safe
    for key, u_val in u.items():
        if key not in d:
            d[key] = inplace(u_val)
            continue

        d_val = d[key]  # key persistent_defaults

        # Case 1: both are dicts → recurse
        if isinstance(d_val, dict) and isinstance(u_val, dict):
            update_nested_dict(d_val, u_val, extend_list=extend_list)

        # Case 1.5: first is list, second is dict
        elif isinstance(d_val, list) and isinstance(u_val, dict):
            if len(d_val) > 0 and isinstance(d_val[0], dict):
                for i in range(len(d_val)):
                    update_nested_dict(d_val[i], u_val, extend_list=extend_list)
            elif len(d_val) > 0:
                d_val.append(u_val)
            else:
                d_val = [u_val]

        # Case 2: both are lists → element-wise merge
        elif isinstance(d_val, list) and isinstance(u_val, list):
            if extend_list:
                d_val.extend(u_val)
            elif inplace_replace_list:
                len_d, len_u = len(d_val), len(u_val)
                if len_u <= len_d:
                    d_val[:len_u] = u_val
                else:
                    d_val = u_val[:len_d]
            else:
                d_val = u_val

        # 3. Fallback: override base value
        else:
            d_val = u_val
        d[key] = inplace(d_val)

    return d


def safe_deepcopy(d, excluded=None, included=None):
    """
    Perform safe deep copy of a dict excluded and included allows to filter out or in entries.
    Both can be either lists or callables

    :param d:
    :param excluded:
    :param included:
    :return:
    """

    if d is None:
        return {}
    copy = {}
    for item in d:
        if excluded is not None:
            if (callable(excluded) and excluded(item)) or (isinstance(excluded, list) and item in excluded):
                continue
        if included is not None:
            if not isinstance(d[item], dict) and (callable(included) and not included(item)) or (isinstance(included, list) and item not in included):
                continue
        if isinstance(d[item], dict):
            copy[item] = safe_deepcopy(d[item], excluded=excluded, included=included)
        else:
            try:
                copy[item] = deepcopy(d[item])
            except:
                try:
                    copy[item] = copy(d[item])
                except:
                    copy[item] = d[item]

    return copy


def get_config_from_dict(field, config={}, default=None, sep='/'):
    '''
    Get config from a dict allowing for recursive search in objects (sep (default slash '/') is used to separate levels)

    :param field:
    :param config:
    :param default:
    :return:
    '''
    fields = field.split(sep)
    value = default
    for i in range(len(fields)):
        field = fields[i]
        if config is not None and field in config:
            if i == len(fields) - 1:
                value = config[field]
            else:
                config = config[field]
        else:
            break

    return value


def _handle_file_name(path, type=None):
    """
    Handle provided path to load a file. In particular the extension.
    :param path:
    :param type:
    :return:
    """
    if type not in ['json', 'pickle', 'dill']:
        raise ValueError('_handle_file_name: type input should be either json, pickle, or dill.')
    typeext = '.'+type
    #
    if path is None or os.path.exists(path):
        return path
    else:
        filename = os.path.basename(path)
        fodler   = os.path.dirname(path)
        path_nox, ext = os.path.splitext(path)
        #
        # No extension has been provided, look for the file with the extension and gzip.
        if not bool(ext):
            if os.path.exists(path_nox+typeext):
                path_out = path_nox+typeext
            elif os.path.exists(path_nox+'.gzip'+typeext):
                path_out = path_nox+'.gzip'+typeext
            elif os.path.exists(path_nox+typeext+'.gzip'):
                path_out = path_nox+typeext+'.gzip'
            elif os.path.exists(path_nox+'.gzip'):
                path_out = path_nox+'.gzip'
            else:
                path_out = path
        # Good extension but file is not found, look for gzip
        elif ext==type:
            if   os.path.exists(path+'.gzip'):
                path_out = path+'.gzip'
            elif os.path.exists(path_nox+'.gzip'+typeext):
                path_out = path_nox+'.gzip'+typeext
            elif os.path.exists(path_nox+'.gzip'):
                path_out = path_nox+'.gzip'
            else:
                path_out = path
        # Wrong extension
        elif type!=ext:
            if os.path.exists(path+typeext):
                path_out = path+typeext
            elif os.path.exists(path_nox+typeext+ext):
                path_out = path_nox+typeext+ext
            elif os.path.exists(path+'.gzip'+typeext):
                path_out = path+'.gzip'+typeext
            elif os.path.exists(path+typeext+'.gzip'):
                path_out = path+typeext+'.gzip'
            elif os.path.exists(path+'.gzip'):
                path_out = path+'.gzip'
            elif os.path.exists(path_nox+typeext):
                path_out = path_nox+typeext
            elif os.path.exists(path_nox+'.gzip'+typeext):
                path_out = path_nox+'.gzip'+typeext
            elif os.path.exists(path_nox+typeext+'.gzip'):
                path_out = path_nox+typeext+'.gzip'
            elif os.path.exists(path_nox+'.gzip'):
                path_out = path_nox+'.gzip'
            else:
                path_out = path
        else:
            path_out = path
        #
        # Return
        if path_out != path:
            print(f'Read {type} file path: {path}, the path has not been found. It has been modified to: {path_out}')
        return path_out


def read_json(path=None, read_metadata=True, quiet=True):
    '''
    Read json as object. Can be a list or a dict

    :param path:
    :return:
    '''
    data = None
    if path is not None:
        if not isinstance(path, list):
            path = [path]
        for p in path:
            p = _handle_file_name(p, type='json')
            try:
                zipped_data = None
                if p.endswith('.zip'):
                    zipped_data = zipfile.ZipFile(p, mode='r')
                    jsonfile = None
                    for file in zipped_data.namelist():
                        if file.endswith('.json') and not file.startswith('.') and not file.startswith('_'):
                            jsonfile = file
                            break
                    p = jsonfile
                    o = zipped_data.open
                else:
                    if p.endswith('.gzip'):
                        o = gzip.open
                    else:
                        o = open
                with o(p, 'r') as json_file:
                    if p.endswith('gzip'):
                        json_bytes = json_file.read()  # 3. bytes (i.e. UTF-8)
                        json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
                        loaded_data = json.loads(json_str)
                    else:
                        loaded_data = json.load(json_file)
                    if isinstance(loaded_data, list):
                        if data is None:
                            data = []
                        data.extend(loaded_data)
                    elif isinstance(loaded_data, dict):
                        if data is None:
                            data = {}
                        data.update(loaded_data)
                if zipped_data is not None:
                    try:
                        zipped_data.close()
                    except:
                        pass
            except Exception as e:
                if not quiet:
                    print('failure when reading data from {}, {}'.format(p, e))
    if data is None:
        data = {}
    if not read_metadata and '__metadata' in data:
        del data['__metadata']
    return data


def read_yaml(path=None, quiet=False):
    data = None
    if path is not None:
        if not isinstance(path, list):
            path = [path]
        data = {}
        for p in path:
            if os.path.exists(f'{p}.yaml'):
                p = f'{p}.yaml'
            elif os.path.exists(f'{p}.yml'):
                p = f'{p}.yml'

            if os.path.exists(p):
                try:
                    with open(p, 'r') as file:
                        dict = yaml.safe_load(file)
                        data.update(dict)
                except Exception as e:
                    if not quiet:
                        print(f'[read_yaml] Failed reading yaml from {path} with {e}')

    return data


def read_toml(path=None, quiet=False):
    data = None
    if path is not None:
        path = make_list(path)
        data = {}
        for p in path:
            try:
                if os.path.exists(f'{p}.toml'):
                    p = f'{p}.toml'
                elif os.path.exists(f'{p}.tml'):
                    p = f'{p}.tml'
                if os.path.exists(p):
                    with open(p, 'rb') as file:
                        dict = tomli.load(file)
                        data.update(dict)
            except Exception as e:
                if not quiet:
                    print(f'[read_toml] Failed reading toml from {p} with {e}')

    return data


def read_dictionary(path=None, types=['json', 'yaml', 'toml'], quiet=True):
    data = {}
    if path is not None:

        if not isinstance(path, list):
            path = [path]
        for path_item in path:
            if 'json' in types:
                out = read_json(path_item, quiet=quiet)
                if out is not None and isinstance(out, dict):
                    data.update(out)

            if 'yaml' in types:
                out = read_yaml(path_item, quiet=quiet)
                if out is not None and isinstance(out, dict):
                    data.update(out)

            if 'toml' in types:
                out = read_toml(path_item, quiet=quiet)
                if out is not None and isinstance(out, dict):
                    data.update(out)

    return data



def dump_json(path=None, data={}, indent=4, sort_keys=False, cls=CustomEncoder, skip_error=True, compressed=False, return_path=False, quiet=False, callables_encoder='pickle'):
    """
    Dump data in json file

    :param path:
    :param data:
    :param indent:
    :param sort_keys:
    :param cls:
    :param skip_error:
    :return:
    """
    if cls == 'default':
        cls = CustomEncoder

    if cls == CustomEncoder and callables_encoder == 'dill':
        cls = CustomEncoderWithDill

    try:
        if path is not None:
            if compressed:
                path = _dump_manage_ext_gzip(path, 'json')
                json_str = dump_json_as_string(data, indent=indent, sort_keys=sort_keys, cls=cls) + "\n"
                json_bytes = json_str.encode('utf-8')
                with gzip.open(path, 'w') as f:
                    f.write(json_bytes)
            else:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=indent, sort_keys=sort_keys, cls=cls)
    except Exception as e:
        if not quiet:
            if skip_error:
                print('failure when writing data to {}, {}'.format(path, e))
            else:
                raise e
    if return_path:
        return path


def dump_json_as_string(data={}, indent=None, sort_keys=False, cls=CustomEncoder, callables_encoder='pickle') -> None:
    """
    Dump data in json string

    :param data:
    :param indent:
    :param sort_keys:
    :param cls:
    :return:
    """
    if cls == 'default' or cls is None:
        cls = CustomEncoder

    if cls == CustomEncoder and callables_encoder == 'dill':
        cls = CustomEncoderWithDill

    try:
        return json.dumps(data, indent=indent, sort_keys=sort_keys, cls=cls)
    except Exception as e:
        return ""


def make_list(obj):
    if not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj


def read_file(path, line_filter=None, top_cnt=0, encoding=None):
    '''
    Read file or files by line

    :param path: String. Path to the file
    :param line_filter: Callable. Filter allowing to exclude lines from the output
    :return:
    '''
    data = None

    paths = make_list(path)
    if paths is not None and len(paths) > 0 :
        data = []
    for path in paths:
        with open(path, 'r', encoding=encoding) as f:
            file_data = f.read().splitlines()

        if line_filter is not None and callable(line_filter):
            clean_data = []
            cnt = 0
            for line in file_data:
                if line_filter(line) and (top_cnt == 0 or cnt < top_cnt ):
                    clean_data.append(line)
                cnt += 1
            file_data = clean_data

        data.extend(file_data)

    return data


def _choose_plt_sys():
    plt_sys = platform.system()
    if plt_sys == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
    if plt_sys == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


def read_pickle(path=None, raise_exception=True):
    """
    Read pickle file
    :param path:
    :param library:
    :return:
    """
    # Handle of pickling library
    if '.cloudpickle' in path:
        pickling_library = cloudpickle
    else:
        pickling_library = pickle
    # Handle list of paths
    if isinstance(path, (list, tuple)):
        out = {}
        for p in path:
            out = update_nested_dict(out, read_pickle(p))
        return out
    # Handle str
    path = _handle_file_name(path, type='pickle')
    _choose_plt_sys()
    out = None
    try:
        if path is not None:
            if 'gzip' in path or path.endswith('gz'):
                with gzip.open(path, 'rb') as file:
                    data = pickling_library.load(file)
                    return data
            else:
                with open(path, 'rb') as f:
                    out = pickling_library.load(f)
    except Exception as e:
        if raise_exception:
            print('failure when reading data from {}, {}'.format(path, e))
    return out


def dump_pickle(path=None, data={}, protocol=pickle.HIGHEST_PROTOCOL, library='pickle', compressed=False, return_path=False, return_file=False) -> None:
    """
    Dump data in a pickle file
    :param path:
    :param data:
    :param protocol:
    :param library: which library is to be used (pickle, cloudpickle)
    :param zip: allow for compressed pickle
    :param split: allow for dividing the data onto splits of specified size
    :return:
    """
    _choose_plt_sys()
    if library == 'pickle':
        pickling_library = pickle
    elif library == 'cloudpickle':
        pickling_library = cloudpickle
    else:
        raise ValueError('You passed the wrong pickling library. Supported classes are pickle and cloudpickle.')
    try:
        if path is not None:
            if compressed:
                path = _dump_manage_ext_gzip(path, 'pickle')
                f = gzip.open(path.format(path), 'wb')
                pickling_library.dump(data, f)
                if not return_file:
                    f.close()
            else:
                f = open(path, 'wb')
                if library == 'pickle':
                    pickling_library.dump(data, f, protocol)
                else:
                    pickling_library.dump(data, f)
                if not return_file:
                    f.close()
        if return_path and return_file:
            return path, f
        elif return_path and not return_file:
            return path
        elif not return_path and return_file:
            return f
    except Exception as e:
        print('failure when writing data to {}, {}'.format(path, e))

def read_dill(path=None):
    """
    Read pickle file using dill (broader scope than pickle)
    :param path:
    :return:
    """
    # Handle list of paths
    if isinstance(path, (list, tuple)):
        out = {}
        for p in path:
            out = update_nested_dict(out, read_dill(p))
        return out
    # Handle a unique path
    path = _handle_file_name(path, type='dill')
    _choose_plt_sys()
    out = None
    try:
        if path is not None:
            if 'gzip' in path or path.endswith('gz'):
                with gzip.open(path, 'rb') as file:
                    data = dill.load(file)
                    return data
            else:
                with open(path, 'rb') as f:
                    out = dill.load(f)
    except Exception as e:
        print('failure when reading data from {}, {}'.format(path, e))
    return out


def dump_dill(path=None, data={}, protocol=pickle.HIGHEST_PROTOCOL, compressed=False, return_path=False, return_file=False) -> None:
    """
    Dump data in a pickle file using dill (broader scope than pickle)
    :param path:
    :param data:
    :param protocol:
    :param zip: allow for compressed pickle
    :param split: allow for dividing the data onto splits of specified size
    :return:
    """
    _choose_plt_sys()
    try:
        if path is not None:
            if compressed:
                path = _dump_manage_ext_gzip(path, 'dill')
                f = gzip.open(path.format(path), 'wb')
                dill.dump(data, f)
                if not return_file:
                    f.close()
            else:
                f = open(path, 'wb')
                dill.dump(data, f, protocol)
                if not return_file:
                    f.close()
    except Exception as e:
        print('failure when writing data to {}, {}'.format(path, e))
    if return_path and return_file:
        return path, f
    elif return_path and not return_file:
        return path
    elif not return_path and return_file:
        return f


def generate_hash_for_data(data="", type='sha256', list_glue='\n', indent=None, sort_keys=False, cls=CustomEncoder, callables_encoder='pickle'):
    """
    Generate hash for the provided data. Sha256 is used by default. If unsupported type is provided, sha256 is used

    The type of data is implicit: that is,
    - the list will be glued with the provided list_glue string
    - the dict will be processed as a json and dumped in a string with the provieded params (indent, sort_keys and cls)

    :param data: data to be hashed
    :param type: String. Type of hash, sha256 by default
    :param list_glue: glue for list-type data
    :param indent: int. indentation for dict-type data
    :param sort_keys: Boolean. Sorting of keys for dict-type data
    :param cls: Class. Custom encoder for dict-type data
    :return:
    """

    supported_types = ['sha256', 'sha512', 'md5']
    if type is None or type not in supported_types:
        type = 'sha256'
    output = data
    if isinstance(output, list):
        output = list_glue.join(output)
    elif isinstance(output, dict):
        output = dump_json_as_string(output, indent=indent, sort_keys=sort_keys, cls=cls, callables_encoder=callables_encoder)
    #
    if isinstance(output, Binary) or isinstance(output, bytes):
        pass
    else:
        output = output.encode()
    #
    if type == 'sha256':
        hash_object = hashlib.sha256(output)
    elif type == 'sha512':
        hash_object = hashlib.sha512(output)
    elif type == 'md5':
        hash_object = hashlib.md5(output)
    #
    data_hex = hash_object.hexdigest()
    return data_hex


def encode_base64(val, encoding='utf-8'):
    """
    Encode a value to a base64 string
    :param val:
    :param encoding
    :return:
    """
    val_encoded = str(val).encode(encoding)
    return base64.b64encode(val_encoded)
