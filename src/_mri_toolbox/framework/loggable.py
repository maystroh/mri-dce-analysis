

from enum import Enum
import _mri_toolbox.utils.system as sysutils
import _mri_toolbox.utils.data as datautils


from _mri_toolbox.utils.loggers import print, logger, setup_logging


class LogType(Enum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class Loggable():





    def get_timestamp(self, force_new=False, stamp_format="%Y%m%d%H%M%S"):
        if not hasattr(self, 'timestamp') or self.timestamp is None or force_new:
            return sysutils.get_timestamp(stamp_format)
        else:
            return self.timestamp



    def init_logger_settings(self, logfilename=None, logfilename_prefix=None, level=None, mode='explicit',
                             track_callers=True, print_console=False, print_file=True, log_format='%(asctime)s %(levelname)s %(message)s', config={}):

        if config is not None and (not hasattr(self, 'logging_config') or not self.logging_config != config):
            '''
            init logging if
            - the config is provided
            - it is different from already applied
            '''
            setup_logging(config, force=True, filename=logfilename)
            self.logging_config = datautils.safe_deepcopy(config)
            self.initialized = True

        if config is None:
            config = {}
        if not hasattr(self, 'timestamp'):
            self.timestamp = self.get_timestamp()
        """
        TODO All of the below is likely to be obsolete. Refactoring is needed
        """
        self.ignore = config.get('ignore', True)
        self.track_callers = config.get('track_callers', track_callers)
        self.print_console = config.get('print_console', print_console)
        self.print_file = config.get('print_file', print_file)
        self.log_handlers = config.get('log_handlers', [{}])
        self.raw_print = config.get('raw_print', False)


    def __init__(self, logfilename=None, logfilename_prefix=None, level=None, mode='implicit', track_callers=True, print_console=False, print_file=True):
        self.init_logger_settings(logfilename, logfilename_prefix, level, mode, track_callers, print_console=print_console, print_file=print_file)

    def is_raw_print(self):
        return hasattr(self, 'raw_print') and self.raw_print





    def modify_logging_configs(self, cfg=None):
        '''
        Apply modification to logging config

        Currently only filenames are supported

        :param cfg:
        :return:
        '''
        if cfg is not None and 'file' in cfg and cfg['file'] is not None and 'filename' in cfg['file']:
            if '%timestamp%' in cfg['file']['filename']:
                cfg['file']['filename'] = cfg['file']['filename'].replace('%timestamp%', self.get_timestamp())
            if '%appname%' in cfg['file']['filename']:
                app_name = type(self).__name__.lower()
                cfg['file']['filename'] = cfg['file']['filename'].replace('%appname%', app_name)
        return cfg

    def get_logging_configs(self, cfg=None):
        '''
        Common method getting logging config

        May be overloaded in child classes

        :param cfg:
        :return:
        '''
        cfg = self.modify_logging_configs()
        return cfg


    def add_log(self, log_text, log_level=LogType.DEBUG):
        if hasattr(self, 'ignore') and self.ignore:
            return
        if not hasattr(self, 'initialized') or not self.initialized:
            self.init_logger_settings(self.get_logging_configs())

        if self.is_raw_print():
            print(log_text)
            return
        elif log_level == LogType.DEBUG:
            return logger.debug(log_text)
        elif log_level == LogType.INFO:
            return logger.info(log_text)
        elif log_level == LogType.WARNING:
            return logger.warning(log_text)
        elif log_level == LogType.ERROR:
            return logger.error(log_text)
        else:
            return Exception("Unhandled logging level")


    def log_info(self, log_text):
        return self.add_log(log_text, LogType.INFO)
