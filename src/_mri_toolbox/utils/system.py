import datetime
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    from importlib_metadata import version, PackageNotFoundError


from subprocess import Popen, PIPE


import inspect


'''
Print business override

use it as follows

from _mri_toolbox.utils.system import builtin_print, print
'''
builtin_print = print
def timestamped_print(*args, **kwargs):
    date_format = kwargs.get('format', '%Y-%m-%d %H:%M:%S.%f')
    builtin_print(f"[{datetime.datetime.now().strftime(date_format)[:-3]}]", *args, **kwargs)

def logprint(*args, **kwargs):
    date_format = kwargs.get('format', '%Y-%m-%d %H:%M:%S.%f')
    builtin_print(f"[{datetime.datetime.now().strftime(date_format)[:-3]}][{inspect.stack()[1][3]}]", *args, **kwargs)


print = timestamped_print


def get_repo_commit_hash(path='./'):
    try:
        process = Popen(["git", "rev-parse", "HEAD"], stdout=PIPE, cwd=path)

        (commit_hash_code, err) = process.communicate()
        exit_code = process.wait()
        commit_hash = commit_hash_code.decode()
        return commit_hash.replace('\n', '')
    except:
        return 'NA'


def get_repo_url(path='./'):
    try:
        process = Popen(["git", "remote", "get-url", "origin"], stdout=PIPE, cwd=path)
        (git_url_code, err) = process.communicate()
        exit_code = process.wait()
        git_url = git_url_code.decode()
        return git_url.replace('\n', '')

    except:
        return 'NA'


def get_package_version(package):
    try:
        __version__ = version(package)
        return(__version__)
    except PackageNotFoundError:
        pass

def get_timestamp(stamp_format="%Y%m%d%H%M%S"):
    '''
    Generate string timestamp using provided format

    :param stamp_format:
    :return:
    '''
    return datetime.datetime.now().strftime(stamp_format)
