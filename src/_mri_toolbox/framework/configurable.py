from _mri_toolbox.framework.loggable import Loggable
import os, platform
import re
from _mri_toolbox.framework.vault import Vault
import _mri_toolbox.utils.data as datautils
import _mri_toolbox.utils.files as filesutils
from _mri_toolbox.utils.loggers import builtin_print, print
try:
    from _mri_toolbox.utils.loggers import context_setup_logging
except:
    pass

class Configurable(Loggable):
    @staticmethod
    def console_output(v):
        return isinstance(v, bool) or (isinstance(v, str) and v == 'console')

    @staticmethod
    def file_output(v):
        if isinstance(v, bool) or (isinstance(v, str) and 'file' in v):

            return True

        return False

    @staticmethod
    def get_file_output_path(v, default, suffix=''):
        if 'file:' in v:
            v = v.replace('file:', '')
            v = f'{v}{suffix}'
            filesutils.get_dir_for_file(v)
            return v
        elif v == 'file':
            v = f'{default}{suffix}'
            return v

    @staticmethod
    def autodoc(doc=True, values=None, unsafe=False, config={}):
        def decorator(func):
            def wrapper(*args, **kwargs):
                self = args[0]  # the instance
                result = func(*args, **kwargs)
                if doc:
                    self.output_autodoc(doc, config={})
                if values:
                    self.output_effective_config(values, config={})
                return result
            return wrapper
        return decorator


    def __init__(self, config={}, parent=None, **kwargs):
        if parent:
            self.timestamp = parent.timestamp
            self.config = parent.config
            self.config_autodoc = parent.config_autodoc
            return
        if not hasattr(self, 'hashable_entries'):
            self.hashable_entries = []
        if config == None:
            config = {}

        if not hasattr(self, 'app_name'):
            if 'app_name' in kwargs:
                self.app_name = kwargs['app_name']
            else:
                self.app_name = type(self).__name__.lower()

        config_paths = []
        config_paths.append(os.path.join(os.path.expanduser("~"), '.herami-properties'))
        config_paths.append(os.path.join(os.getcwd(), '.herami-properties'))
        config_paths.append(os.path.join(os.getcwd(), f'{self.app_name}.properties'))
        try:
            nodename = platform.uname().node.lower()
            config_paths.append(os.path.join(os.getcwd(), f'.{nodename}.herami-properties'))
            config_paths.append(os.path.join(os.getcwd(), f'{nodename}.properties'))
            config_paths.append(os.path.join(os.getcwd(), f'{self.app_name}.{nodename}.properties'))
        except Exception as e:
            print(f'Failed customizing config {e}')

        loaded_configs = []
        for config_path in config_paths:
            loaded_configs.append(datautils.read_dictionary(config_path))

        self.default_config = self.generate_default_config()
        cfg = {}
        cfg = self.update_nested_dict(cfg, self.default_config)
        for loaded_config in loaded_configs:
            cfg = self.update_nested_dict(cfg, loaded_config)
        if cfg is not None and cfg.get('override_py_config', False):
            cfg = self.update_nested_dict(config, cfg)
        else:
            cfg = self.update_nested_dict(cfg, config)
        config = cfg
        # if self.default_config:
        #     for entry in self.default_config:
        #         if entry not in config:
        #             config[entry] = self.default_config[entry]
        self.config = config
        persistent_defaults = self.config.setdefault('persistent_defaults', False)
        if persistent_defaults:
            self.effective_config = config
        else:
            self.effective_config = {}
            self.effective_config = self.update_nested_dict(self.effective_config, {})
        self.adapt_config()

        log_config = {}
        try:
            log_path = os.path.join(os.getcwd(), f'log.properties')
            if os.path.exists(log_path):
                log_config = datautils.read_dictionary(log_path)
        except Exception as e:
            print(f'Failed getting log configs with {e}')

        self.init_logger_settings(config=self.get_logging_configs(log_config), logfilename=f'{self.app_name}.log')
        self.put_config_in_vault()
        self.init_timestamp = self.get_timestamp()
        self.print = print
        self.pyprint = builtin_print

    def get_logging_configs(self, default_cfg=None):
        cfg = self.get_config('logging', default_cfg)
        cfg = self.modify_logging_configs(cfg)
        return cfg

    def is_raw_print(self):
        return self.get_config('is_log_print', False)

    def generate_default_config(self):
        return {}

    def update_nested_dict(self, ref, upd, **kwargs):
        return datautils.update_nested_dict(ref, upd, **kwargs)


    def is_quiet(self):
        if not hasattr(self, 'quiet'):
            self.quiet = self.get_config('quiet', False)
        return self.quiet

    def safe_deepcopy(self, d, excluded=None, included=None):
        return datautils.safe_deepcopy(d, excluded, included)

    def safe_field(self, item):
        item = str(item)
        entry = (item or '').lower()
        return not ('password' in entry or 'token' in entry or 'api' in entry or 'db_uri' in entry or (hasattr(self, 'hashable_entries') and entry in self.hashable_entries))

    def unsafe_field(self, item):
        item = str(item)
        entry = (item or '').lower()
        if ('password' in entry or 'token' in entry or 'api' in entry or 'db_uri' in entry or (hasattr(self, 'hashable_entries') and entry in self.hashable_entries)):
            return True
        return False

    def put_config_in_vault(self):
        if self.get_config('safe_config', True):
            if not hasattr(self, 'vault'):
                self.vault = Vault()

            safe_config = self.safe_deepcopy(self.config, included=self.safe_field)
            sensitive_config = self.safe_deepcopy(self.config, included=self.unsafe_field)

            self.config = safe_config
            self.vault.update(sensitive_config)


    def adapt_config(self):
        '''
        In-place config modifier
        :return:
        '''
        return self.config



    def get_config(self, field, default=None, allow_env=False, desired_config=None, comment=None):

        if field[0]=='/':
            field = field[1:]
        if field[-1]=='/':
            field = field[:-1]

        if desired_config is not None:
            cfg = desired_config
        else:
            cfg = {}
            if hasattr(self, 'config'):
                cfg = self.update_nested_dict(cfg, self.config)
            if hasattr(self, 'vault'):
                cfg = self.update_nested_dict(cfg, self.vault)


        if hasattr(self, 'config') and cfg is not None:
            v = None
            if 'cfg_profile' in cfg and cfg['cfg_profile'] is not None:
                try:
                    v = self.get_config_from_dict(field, cfg[cfg['cfg_profile']], None)
                except Exception:
                    pass
            if v is None:
                v = self.get_config_from_dict(field, cfg, None)

        if not (hasattr(self, 'config') and cfg is not None) or (v is None and allow_env):
            v = self.get_env_var(field)
            if v is None:
                v = default

        if v is None and '{{' in field:
            v = field

        if v is not None and isinstance(v, str) and '{{' in v:
            entries = re.findall(r"{{([A-Za-z0-9_]+)}}", v)
            if entries is not None and len(entries) > 0:
                for entry in entries:
                    elts = entry.split(':')
                    default_value = entry if len(elts) == 1 else elts[1]
                    cfg_name = elts[0]
                    v = v.replace(f'{{{{{entry}}}}}', self.get_env_var(cfg_name, default=self.get_config(cfg_name, default_value)))

        if v is None:
            v = default

        try:
            self.set_effective_config(field, v)
        except Exception:
            pass

        try:
            self.autodoc_config(field, v, default, comment)
        except:
            pass

        return v

    def output_effective_config(self, flavor, config={}):
        try:
            if Configurable.console_output(flavor):
                print("Effective config after app execution")
                print(datautils.dump_json_as_string(self.safe_deepcopy(self.effective_config), indent=4))
            elif Configurable.file_output(flavor):
                f = Configurable.get_file_output_path(flavor,
                                                      f'{self.app_name}-effective_config-{self.get_timestamp()}.json')
                datautils.dump_json(path=f, data=self.safe_deepcopy(self.effective_config), sort_keys=(config or {}).get('sort_keys', True))
        except Exception as e:
            print(f'Failed printing effective config with {e}')

    def output_autodoc(self, flavor='print', config={}):
        try:
            if Configurable.console_output(flavor):
                print("Config params used in the app")
                print(datautils.dump_json_as_string(self.config_autodoc, indent=4))
            elif Configurable.file_output(flavor):
                f = Configurable.get_file_output_path(flavor, f'{self.app_name}-config_autodoc', '_raw.json')
                datautils.dump_json(path=f, data=self.config_autodoc, sort_keys=(config or {}).get('sort_keys', True))
                f = Configurable.get_file_output_path(flavor, f'{self.app_name}-config_autodoc', '_clean.json')
                datautils.dump_json(path=f, data=self.clean_autodoc(), sort_keys=(config or {}).get('sort_keys', True))
        except Exception as e:
            print(f'Failed printing config autodoc with {e}')

    def clean_autodoc(self):
        clean_autodoc = {}
        if hasattr(self, 'config_autodoc'):
            for key, item in self.config_autodoc.items():
                if 'comment' in item and item['comment']:
                    clean_autodoc[key] = item

        return clean_autodoc

    def autodoc_config(self, field, v, default, comment=None):

        if not hasattr(self, 'config_autodoc'):
            self.config_autodoc = {}
        val_desc = {'comment': comment, 'default': default}
        b64key = datautils.encode_base64(val_desc)
        code = f'{field}::{b64key}'
        self.config_autodoc[code] = {'comment': comment, 'default': default}

    def get_env_var(self, var, default=None):
        return os.environ.get(var, default)


    def get_config_from_dict(self, field, config={}, default=None, sep='/'):
        """
        Get config from a dict allowing for recursive search in objects (slash '/' is used to separate levels)

        :param field: Field to be retrieved.
        :param config: dictionary to search the fied in
        :param default: Default value to be returned if the field is not found
        :return:
        """
        return datautils.get_config_from_dict(field, config=config, default=default, sep=sep)


    def set_effective_config(self, field, value):
        # nathan: this function is probably not behaving as intended
        #         use preferably set_default_config defined above
        fields = field.split("/")
        conf = self.effective_config

        for i in range(len(fields)):
            field = fields[i]
            if field in conf:
                conf = conf[field]
            elif field not in conf:
                c = value
                for j in range(len(fields)-2, i-1, -1):
                    c = {field: c}
                conf[field] = c
