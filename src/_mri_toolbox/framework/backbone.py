


import os
import sys
import inspect
from copy import deepcopy

try:
    import pandas as pd
except:
    print('Failed importing pandas. Dataframe stuff will not work')
#
from _mri_toolbox.utils.data import dump_pickle, dump_dill
import _mri_toolbox.utils.system as sysutils
import _mri_toolbox.utils.data as datautils
import _mri_toolbox.utils.files as fileutils
from _mri_toolbox.framework.configurable import Configurable

from _mri_toolbox.utils.loggers import builtin_print, print


class Backbone(Configurable):


    def __init__(self, config=None, **kwargs):
        super(Backbone, self).__init__(config, **kwargs)

        self.sysutils = sysutils if kwargs.get('load_sysutils', True) else None # this variable make the class unpickable... use False to be able to pickle a Backbone object
        if self.get_config('config_file', None):
            config_files = self.make_list(self.get_config('config_file'))
            for config_file in config_files:
                self.config = self.update_nested_dict(self.config, self.read_dictionary(config_file, quiet=False))
            self.config.pop('config_file')
        #
        section = self.get_config('section', None)
        if section is not None and section in self.config:
            self.config = self.get_config(section)
        #
        if self.get_config('stringified_cfg', None):
            configs = self.make_list(self.get_config('stringified_cfg'))
            for config in configs:
                self.config = self.update_nested_dict(self.config, self.str2obj(config))

        # If at this stage the logging config is provided, try to init it. Maybe it is different from what is already applied
        if self.get_config('logging', None) is not None:
            self.init_logger_settings(config=self.get_logging_configs(None))
        self.post_init()
        self.run_inits()

    def post_init(self):
        pass

    def run_inits(self):
        run_inits = self.get_config('run_inits', True)
        if run_inits:
            try:
                object_methods = []
                for method_name in dir(self):
                    try:
                        if (callable(
                            getattr(self, method_name)) and 'op_init_' in method_name and not method_name.startswith(
                            '__init_')):
                            object_methods.append(method_name)
                    except Exception as e:
                        print(f'Failed processing {method_name} to init {e}')
                for object_method in object_methods:
                    try:
                        getattr(self, object_method)()
                    except Exception as e:
                        print(f'Failed running {object_method} with {e}')

            except Exception as e:
                print(f'Failed getting objects to init {e}')


    def make_list(self, obj, return_converted=False):
        converted = False
        if not isinstance(obj, (list, tuple)):
            obj = [obj]
            converted = True
        if return_converted:
            return obj, converted
        else:
            return obj


    def clean_dict(self, dic, keys, default=None, explicit=False):
        clean_dict = {}
        for k in keys:
            if k in dic or explicit:
                clean_dict[k] = dic.get(k, default)

        return clean_dict





    def clean_sensitive_data(self, config):
        '''
        Clean sensitive data to avoid prining passwords in logs
        :param config:
        :return:
        '''
        if not hasattr(self, 'hashable_entries'):
            self.hashable_entries = ['DB_URI', 'DB_URI_OPS']

        safe_config = {}
        safe_config = self.update_nested_dict(safe_config, config, copy=True)
        def clean_dict(cfg):
            for entry in cfg:
                if isinstance(cfg[entry], dict):
                    cfg[entry] = clean_dict(cfg[entry])
                else:
                    if 'password' in entry or 'token' in entry or 'DB_URI' in entry or entry in self.hashable_entries:
                        cfg[entry] = self.generate_hash_for_string(cfg[entry])
            return cfg
        safe_config = clean_dict(safe_config)
        return safe_config

    def get_default_metadata(self):
        curframe = inspect.currentframe()

        calframe = inspect.getouterframes(curframe, 2)
        caller = {}

        for i in range(len(calframe)):
            called_script = calframe[i][1]
            if 'backbone.py' not in called_script:
                caller['script'] = called_script
                caller['method'] = os.path.basename(calframe[i][3])
                break

        supported_modules = ['artifactor', 'data_mover', 'dataops_orchestrator', 'dicom_cleaner',
                             'performance_evaluator', 'tensorflower1x', 'tools', ]
        explicit_modules = self.get_config('explicit_modules', [])
        supported_modules.extend(explicit_modules)
        metadata = {}
        imported_modules = sys.modules.keys()
        for module in supported_modules:
            if module in imported_modules or module in explicit_modules:
                metadata['{}.version'.format(module)] = sysutils.get_package_version(module)

        metadata['caller'] = caller
        metadata['timestamp'] = self.get_timestamp()
        try:
            metadata['config'] = self.safe_deepcopy(self.config)
        except:
            self.dump_dill()
        try:
            metadata['effective_config'] = self.safe_deepcopy(self.effective_config)
        except:
            pass
        metadata['commit'] = self.get_repo_commit_hash(path='./')
        metadata['git_url'] = self.get_repo_url(path='./')
        return metadata


    def extract_uid_from_path(self, path, suffix=None, prefix=False, prefix_glue='_'):
        return fileutils.extract_uid_from_path(path, suffix, prefix=prefix, prefix_glue=prefix_glue)


    def dump_json(self, path, data, indent=4, sort_keys=True, metadata=None, with_metadata=False, keep_history=True, skip_error=True, cls='default', compressed=False, timestamped=False, safe=False, return_path=False, quiet=False):
        meta = metadata
        if with_metadata:
            if meta is None:
                meta = {}
            meta.update(self.get_default_metadata())
            if metadata is not None:
                meta.update(metadata)
        if safe:
            meta = self.clean_sensitive_data(meta)
        if meta is not None:
            history = None
            if '__metadata' in data:
                previous_metadata = deepcopy(data['__metadata'])
                if 'history' in previous_metadata:
                    history = deepcopy(previous_metadata['history'])
                    del previous_metadata['history']
                else:
                    history = {}
                history[str(self.get_timestamp())] = previous_metadata

            data['__metadata'] = meta
            if history is not None and keep_history:
                data['__metadata']['history'] = history
        self.get_dir_for_file(path)

        if timestamped and path is not None:
            if path.endswith('.json'):
                path = path.replace('.json', f'-{self.get_timestamp()}.json')
            else:
                path = f'{path}-{self.get_timestamp()}'

        return datautils.dump_json(path, data, indent=indent, sort_keys=sort_keys, skip_error=skip_error, cls=cls, compressed=compressed, return_path=return_path, quiet=quiet)

    def dump_json_as_string(self, data, indent=4, sort_keys=True, cls='default'):
        return datautils.dump_json_as_string(data, indent=indent, sort_keys=sort_keys, cls=cls)


    def read_json(self, path=None, read_metadata=True, quiet=True):
        return datautils.read_json(path, read_metadata=read_metadata, quiet=quiet)

    def read_dictionary(self, path=None, types=['json', 'yaml', 'toml'], quiet=True):
        return datautils.read_dictionary(path, types=types, quiet=quiet)


    def read_file_by_line(self, path, line_filter=None, top_cnt=0, encoding=None):
        '''
        Read file or files by line

        :param path: String. Path to the file
        :param line_filter: Callable. Filter allowing to exclude lines from the output
        :return:
        '''
        return datautils.read_file(path, line_filter=line_filter, top_cnt=top_cnt, encoding=encoding)


    def mkdirs(self, dirname):
        fileutils.mkdirs(dirname)


    def str2obj(self, str_obj):
        return datautils.str2obj(str_obj)


    def get_mapped_paths(self, root_path, filter=None, filename_suffix=None, mode='uid', entry_key_modifier=None, custom_modifier=None, glue='/', keep_one=False):
        """
        Generate a map with keys being uids and the entries being paths

        :param paths: a path or a list of paths to be scanned for files
        :param suffix: explicit suffix to remove from filename to get the uid value
        :return: the dictionary of paths per uid
        """
        return self.get_filenames_to_path_map(root_path=root_path, filter=filter, filename_suffix=filename_suffix,
                                              mode=mode, entry_key_modifier=entry_key_modifier, custom_modifier=custom_modifier, glue=glue, keep_one=keep_one)

    def get_filenames_to_path_map(self, root_path, filter=None, filename_suffix=None, mode='raw', files_only=True, dirs_only=False, entry_key_modifier=None, custom_modifier=None, glue='/', keep_one=False, debug=False):
        """
        Generate a map with keys being names of files or uids and the entries being paths

        :param paths: a path or a list of paths to be scanned for files
        :param suffix: explicit suffix to remove from filename to get the uid value

        :param mode:    mode of scanning files. Two modes are supported:
                        - raw (by default), setting filename as a key;
                        - uid (by default), setting uid (as instanceUid) as a key;
        :param files_only Boolean. Map only files
        :param dirs_only Boolean. Map only dirs
        :return: the dictionary of paths per uid
        """
        if not isinstance(root_path, list):
            root_path = [root_path]
        mapped_paths = {}
        for root in root_path:
            if os.path.isfile(root):
                if (filter is not None and callable(filter) and filter(root)) or (filter is None):
                    paths = [root, ]
            else:
                paths = fileutils.get_paths_recursively(root, filter, files_only=files_only, dirs_only=dirs_only, debug=debug)
            mapped = fileutils.map_paths_by_filename(paths, filename_suffix, mode=mode, entry_key_modifier=entry_key_modifier, custom_modifier=custom_modifier, glue=glue, keep_one=keep_one)
            for uid in mapped:
                if uid in mapped_paths and not keep_one:
                    if not isinstance(mapped_paths[uid], list):
                        mapped_paths[uid] = [mapped_paths[uid]]
                    if isinstance(mapped[uid], list):
                        mapped_paths[uid].extend(mapped[uid])
                    else:
                        mapped_paths[uid].append(mapped[uid])
                else:
                    mapped_paths[uid] = mapped[uid]

        return mapped_paths

    def get_paths_recursively(self, root_path, files_filter=None, files_only=True, dirs_only=False, depth=-1, explicit_depth=False):
        paths = fileutils.get_paths_recursively(root_path, files_filter, files_only=files_only, dirs_only=dirs_only, depth=depth, explicit_depth=explicit_depth)
        return paths


    def get_package_version(self, package_name):
        return sysutils.get_package_version(package_name)

    def get_repo_commit_hash(self, path='./'):
        return sysutils.get_repo_commit_hash(path)

    def get_repo_url(self, path='./'):
        return sysutils.get_repo_url(path)


    def get_dir_for_file(self, filename, walk_dir=None, dest_dir=None):
        return fileutils.get_dir_for_file(filename, walk_dir, dest_dir)

    def dump_pickle_or_dill(self, data={}, filename=None, dest='./', timestamped=True, compressed=True, type='pickle', return_path=False, ext=None, return_file=False):
        if filename is None:
            filename = self.get_config('export_filename', 'export')
        if timestamped:
            filename = '{}-{}'.format(filename, self.get_timestamp())
        if ext is None:  # or ('gzip' not in ext and compressed):
            if compressed:
                ext = 'gzip'
            else:
                ext = type
            if filename[-len(ext) - 1:] == '.' + ext:
                filename = filename[:-len(ext) - 1]
        elif ext is not None and 'gzip' not in ext and compressed:
            ext = ext.replace('.', '') + '.gzip'
        elif ext is not None:
            ext = ext.replace('.', '')
        #
        # Get function
        if type not in ['pickle', 'dill']:
            raise ValueError(f'dump_pickle_or_dill does not support the type: {type}')
        dump_fct = dump_dill if type == 'dill' else dump_pickle
        #
        if isinstance(data, list) and len(data) > 1:
            cnt = 0
            outs = list()
            for sample in data:
                cnt += 1
                sample['timestamps'] = {'launch': self.timestamp, 'export': self.get_timestamp(True)}
                path = os.path.join(dest, '{}.{}.{}'.format(filename, cnt, ext))
                fileutils.get_dir_for_file(path)  # mkdir
                path = dump_fct(path=path, data=sample, compressed=compressed, return_path=return_path, return_file=return_file)
                outs.append(path)
            return outs
        else:
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            if timestamped:
                data['timestamps'] = {'launch': self.timestamp, 'export': self.get_timestamp(True)}
            path = os.path.join(dest, '{}.{}'.format(filename, ext))
            fileutils.get_dir_for_file(path)  # mkdir
            return dump_fct(path=path, data=data, compressed=compressed, return_path=return_path, return_file=return_file)

    def dump_pickle(self, data={}, filename=None, dest='./', timestamped=True, compressed=True, return_path=False):
        return self.dump_pickle_or_dill(data=data, filename=filename, dest=dest, timestamped=timestamped, compressed=compressed, type='pickle', return_path=return_path)

    def dump_dill(self, data={}, filename=None, dest='./', timestamped=True, compressed=True):
        return self.dump_pickle_or_dill(data=data, filename=filename, dest=dest, timestamped=timestamped, compressed=compressed, type='dill')

    def read_pickle_or_dill(self, path=None, type='pickle'):
        if type == 'pickle':
            return datautils.read_pickle(path)
        elif type == 'dill':
            return datautils.read_dill(path)
        else:
            raise ValueError(f'read_pickle_or_dill does not support the type: {type}')

    def read_pickle(self, path=None):
        return self.read_pickle_or_dill(path, type='pickle')

    def read_dill(self, path=None):
        return self.read_pickle_or_dill(path, type='dill')

    def read_csv(self, path=None):
        return self.read_file_by_line(path)

    def read_file(self, path):
        """
        Read a file with an unknown extension. Supported format are json, pickle, and dill.
        """
        if '.json' in path:
            return self.read_json(path, quiet=False)
        elif ('.pickle' in path) or ('.pkl' in path):
            return self.read_pickle(path)
        elif ('.dill' in path):
            return self.read_dill(path)
        elif '.csv' in path:
            return self.read_csv(path)
        else:
            raise ValueError(f'The format of the file {path} could not be automatically determined.')


    def generate_hash_for_data(self, data, type='sha256', cls=datautils.CustomEncoder, callables_encoder='pickle', sort_keys=False):
        """
        Facility method for fileutils.generate_hash_for_data

        :param data: String. Data to generate hash for
        :param type: String. Type of hash, sha256 by default
        :return:
        """
        return datautils.generate_hash_for_data(data=data, type=type, cls=cls, callables_encoder=callables_encoder, sort_keys=sort_keys)

    def generate_hash_for_string(self, data="", type='sha256', list_glue='\n', indent=None, sort_keys=False, cls=datautils.CustomEncoder):
        """
        Facility method for datautils.generate_hash_for_data

        :param data: data to be hashed
        :param type: String. Type of hash, sha256 by default
        :param list_glue: glue for list-type data
        :param indent: int. indentation for dict-type data
        :param sort_keys: Boolean. Sorting of keys for dict-type data
        :param cls: Class. Custom encoder for dict-type data
        :return:
        """
        return datautils.generate_hash_for_data(data=data, type=type, list_glue=list_glue, indent=indent, sort_keys=sort_keys, cls=cls)









    def update_nested_dict(self, d, u, extend_list=False, copy=False, *args, **kwargs):
        """
        Update a nested dictionary.
        :param d [dict]: Dictionary to be updated.
        :param u [dict]: Update dictionary.
        :param extend_list [bool]: If d already contains a value to be updated and this latter is a list, extend the original list.
        :return:
        """
        return datautils.update_nested_dict(d, u, extend_list=extend_list, copy=copy, *args, **kwargs)






    def encode_base64(self, val, encoding='utf-8'):
        """
        Encode a value to a base64 string
        :param val:
        :param encoding
        :return:
        """
        return datautils.encode_base64(val, encoding)


# TEST
