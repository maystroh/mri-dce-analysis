import argparse
import re
import sys
import json
from _mri_toolbox.framework.backbone import Backbone

from _mri_toolbox.utils.loggers import builtin_print, print


class Argumentable(Backbone):

    def __init__(self, config=None, **kwargs):
        """
        The Argumentable class allow for specifying configurations in command line. The order of load is as follows
        - The config specified in python file at class' creation
        - The config from one or several json files
        - The config from --config object

        The later updates the earlier

        :param config:
        :param kwargs:
        """
        super(Argumentable, self).__init__(config=config, **kwargs)
        self.name = kwargs.get('name', 'argumentable script')
        self.args = self.get_args()
        #
        if self.args.timestamp is not None:
            self.timestamp = self.args.timestamp
        #
        # section used when calling a giant config but we are only interested by subpart, e.g., the mlflow config section of the evaluation pipeline (DataKitchen config part is useless in this case)
        section = self.get_config('section', None)
        if self.args.section is not None:
            section = self.args.section

        if self.args.master_json_file_config is not None:
            self.config = {}
            config_files = self.make_list(self.args.master_json_file_config)
            for config_file in config_files:
                self.config = self.update_nested_dict(self.config, self.read_dictionary(config_file))

        elif self.args.json_file_config is not None or self.args.config_file is not None :
            config_files = []
            if self.args.json_file_config:
                config_files.extend(self.make_list(self.args.json_file_config))
            if self.args.config_file:
                config_files.extend(self.make_list(self.args.config_file))
            for config_file in config_files:
                d = self.read_dictionary(config_file)
                #
                if section is not None:
                    d = self.get_config(section, d, desired_config=d)
                self.config = self.update_nested_dict(self.config, d, extend_list=True)
        self.config.pop('config_file') if 'config_file' in self.config else None

        if self.args.unique_config is not None and len(self.args.unique_config) > 0:
            self.config = self.args.unique_config
            self.args.unique_config = None

        elif self.args.config is not None:
            self.config = self.update_nested_dict(self.config, self.args.config, extend_list=True)
            self.args.config = None

        if self.get_config('logging', None) is not None:
            # If at this stage the logging config is provided, try to init it. Maybe it is different from what is already applied
            self.init_logger_settings(config=self.get_logging_configs(None))

        if 'master_config' in self.config:
            self.config = self.update_nested_dict(self.config, self.config['master_config'])

        try:
            print_cfg = self.get_config('print_cfg', True)
            if print_cfg and not self.is_quiet():
                print('-----------------------')
                print(type(self).__name__)
                print('--- Loaded config -----')
                safe_config = self.clean_sensitive_data(self.config)
                print(self.dump_json_as_string(safe_config))
                print('-----------------------')
            if self.get_config('logging', None) is not None and len(self.get_config('logging', None)) > 0:
                self.log_info('-----------------------')
                self.log_info(type(self).__name__)
                self.log_info('--- Loaded config -----')
                safe_config = self.clean_sensitive_data(self.config)
                self.log_info(self.dump_json_as_string(safe_config))
                self.log_info('-----------------------')
        except Exception as e:
            print(f'Failed printing config with {e}')

    def get_name(self):
        return self.name

    def get_args(self):
        parser = argparse.ArgumentParser(
            prog=self.get_name(),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )

        def convert_arg_line_to_args(arg_line):
            """ Function to replace the ArgumentParser method which handles a single argument line from a file.
            Handle multiple values on one line in args file, and skip lines containing comments or blank lines. """
            # Check for lines with whitespace (optional) and then a comment character (#), OR just lines with blank space.
            if (re.match(r'^[\s]*#', arg_line) or re.match(r'^[\s]*$', arg_line)):
                return []  # Return nothing (argparse doesn't handle None well)
            elif arg_line.endswith('}'):
                elts = arg_line.split()
                val = arg_line.replace(elts[0]+' ', '')
                val = val.strip()
                return [elts[0], val]
            else:
                # Split the line based on whitespace.
                return arg_line.split()

        parser.convert_arg_line_to_args = convert_arg_line_to_args

        def update_dict():
            class dict_update_action(argparse.Action):
                def __call__(self, parser, args, values, option_string=None):
                    try:
                        if hasattr(args, self.dest):
                            val = getattr(args, self.dest)
                            val.update(values)
                            setattr(args, self.dest, val)
                    except:
                        setattr(args, self.dest, values)
            return dict_update_action

        parser.add_argument('-c', '--config', type=self.str2obj, default={}, help='config dict', action=update_dict())
        parser.add_argument('-u', '--unique_config', type=self.str2obj, default={}, help='unique args config ignoring everything else')
        parser.add_argument('--timestamp', default=None, help='force self.timestamp to a specific value')
        parser.add_argument('-j', '--json_file_config', type=str, nargs='+', default=None, help='list of config files (json, yaml, toml)')
        parser.add_argument('-f', '--config_file', type=str, nargs='+', default=None, help='list of config files (json, yaml, toml)')
        parser.add_argument('-s', '--section', type=str, default=None, help='specify a section of config')
        parser.add_argument('-m', '--master_json_file_config', type=str, nargs='+', default=None, help='config')
        parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbose')
        parser = self.overload_parser(parser)
        args, unknown_args = parser.parse_known_args(sys.argv[1:])
        del sys.argv[1:]  # remove parsed cmd that are sometimes reparse when wrong argumentable call

        if args.verbose:
            print("\n\n====\n\n")
            print("\n\nUnknown arguments:")
            print(unknown_args)
            print("\n\n====\n\n")

            print("\n\nArgs")
            print(json.dumps(vars(args), sort_keys=True, indent=4))
            print("\n\n====\n\n")

        return args

    def overload_parser(self, parser):
        return parser
