from _mri_toolbox.utils.logging import builtin_print

import logging

import _mri_toolbox.utils.data as datautils

import sys
import os

global logging_initialized

logging_initialized = False


formatters = {}


levels = {
    'critical': logging.CRITICAL,
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'notset': logging.NOTSET,
}


def get_level(code='info'):
    return levels.get(code, code if isinstance(code, int) else logging.INFO)


def get_format(code='default'):
    if code is None or code == '' or code == 'default':
        return "%(asctime)s [%(threadName)-12.12s][%(filename)-20.20s][%(funcName)-12.12s][%(levelname)-5.5s]  %(message)s"
    else:
        return code


def context_setup_logging(force=False, filename="output.log"):
    log_path = os.path.join(os.getcwd(), f'log.properties')
    if os.path.exists(log_path):
        log_config = datautils.read_dictionary(log_path)
        setup_logging(log_config, force=force, filename=filename)


def setup_logging(config={}, force=False, filename="output.log"):
    '''
    Setup logging to be called to initialize the logging config

    By default the logging is done on info level in console. The file config can be provided if needed.
    :param config:
    :param force:
    :return:
    '''
    global logging_initialized
    if not logging_initialized or force:
        fileconfig = config.get('file', None)
        consoleconfig = config.get('console', {})
        rootLogger = logging.getLogger()
        rootLogger.handlers.clear()
        rootLogger.setLevel(get_level(config.get('level', 'debug')))

        if fileconfig is not None:
            filename = fileconfig.get('filename', filename)
            fileLogFormatter = logging.Formatter(get_format(fileconfig.get('format')))
            if fileconfig.get('rotation', {}).get('type', 'time') == 'time':
                fileHandler = logging.handlers.TimedRotatingFileHandler(filename=filename, when=fileconfig.get('rotation', {}).get('when', 'midnight'), backupCount=fileconfig.get('rotation', {}).get('backupCount'))
            else:
                fileHandler = logging.FileHandler(filename)
            fileHandler.setLevel(get_level(fileconfig.get('level')))
            fileHandler.setFormatter(fileLogFormatter)
            rootLogger.addHandler(fileHandler)

        if consoleconfig is not None:
            consoleLogFormatter = logging.Formatter(get_format(consoleconfig.get('format')))
            stream = sys.stdout
            if consoleconfig.get('stream', 'out') == 'err':
                stream = sys.stderr

            consoleHandler = logging.StreamHandler(stream)
            consoleHandler.setFormatter(consoleLogFormatter)
            consoleHandler.setLevel(get_level(consoleconfig.get('level')))
            rootLogger.addHandler(consoleHandler)

        logging_initialized = True


logger = logging.getLogger()

'''
print* shortcuts to allow various logging levels

Print may be imported in any python module to enable logging support
'''
print = logger.info
printd = logger.debug
printi = logger.info
printw = logger.warning
printe = logger.error
