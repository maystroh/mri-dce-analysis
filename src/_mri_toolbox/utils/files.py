from _mri_toolbox.utils.logging import print
from _mri_toolbox.utils.exceptions import BreakAllLoops
import os
import numpy as np

import re


try:
    import pyminizip
except:
    pass

import os
import _mri_toolbox.utils.data as datautils

# Lambda functions
is_regex_pattern = lambda reg: reg is not None and bool(re.search(r'[\.\^\$\*\+\?\{\}\[\]\|\(\)]', reg))
find_first_regex_dir = lambda reg: re.split(r'[*?[]', reg, 1)[0] if is_regex_pattern(reg) else reg
remove_regex_characters = lambda reg: re.compile(r'[\\^$.|?*+(){}[\]!/]').sub('', reg) if reg is not None else reg


# Functions
def make_list(obj):
    if not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj


def get_dir_for_file(file, walk_dir=None, dest_dir=None):
    if dest_dir is not None and walk_dir is not None:
        dest_file = file.replace(walk_dir, dest_dir)
    else:
        dest_file = file
    dest_dir = os.path.dirname(dest_file)
    mkdirs(dest_dir)
    return dest_file


def extract_uid_from_path(path, suffix=None, prefix=False, prefix_glue='_'):
    exts = ['.dcm', '.png', '.jpg', '.bmp', '.tiff']
    try:
        uid = os.path.basename(path)
        for ext in exts:
            uid = uid.replace(ext, '')
        if suffix is not None:
            suffix = make_list(suffix)
            for suff in suffix:
                uid = uid.replace(suff, '')

        if prefix_glue in uid and prefix:
            elts = uid.split(prefix_glue)
            uid = elts[-1]
        return uid
    except:
        print('[extract_uid_from_path] Failed getting uid from {}'.format(path))

    return None


def mkdirs(dirname):
    '''
    Create all required dirs for a given path
    :param dirname:
    :return:
    '''
    try:
        os.makedirs(dirname, exist_ok=True)
    except Exception as e:
        print('Exception raise on mkdirs as {}'.format(e))


def filter_files_by_criteria(filename, criteria):
    def lookup_in_list(name, items):
        for item in items:
            if re.match(item, filename):
                return True
    if isinstance(criteria, dict):
        inculde = criteria.get('include', [])
        exclude = criteria.get('exclude', [])
        return lookup_in_list(filename, inculde) and not lookup_in_list(filename, exclude)

    elif isinstance(criteria, (str, list)):
        criteria_list = make_list(criteria)
        return lookup_in_list(filename, criteria_list)
    return False

# allow to convert a path regexed such as "/home/*/data/*dcm" to a list of paths of all existing "dcm" files for all users


def get_paths_recursively(src_root_path, files_filter=None, files_only=True, dirs_only=False, depth=-1, banned_strings=None, debug=False, explicit_depth=False):
    '''
    Recursively get files paths provided root path and a filter: by default, hidden files (.) are skipped

    :param src_root_path:
    :param files_filter:
    :return:
    '''
    # debug config to found x file maximum
    file_limit = np.inf
    file_index_found = []
    if isinstance(debug, str):
        debug = datautils.compile_function_from_string(debug)
    #
    if isinstance(debug, (np.ndarray, list)):
        file_index_found = debug
    elif debug == True:
        file_limit = 100
    elif debug != False and isinstance(debug, (int, float)):
        file_limit = int(debug)
    #
    files = []
    cnt_accepted = 0
    if src_root_path is not None:
        try:
            src_root_paths = make_list(src_root_path)
            for src_root_path in src_root_paths:
                if src_root_path.endswith('/'):
                    src_root_path = src_root_path[:-1]
                for root, directories, filenames in os.walk(src_root_path):
                    entries = []
                    if files_only:
                        entries.extend(filenames)
                    elif not files_only and dirs_only:
                        entries.extend(directories)
                    elif not files_only and not dirs_only:
                        entries.extend(filenames)
                        entries.extend(directories)
                    for filename in entries:
                        full_file_name = os.path.join(root, filename)
                        if (((os.path.isfile(full_file_name) and files_only) or not files_only) and ((os.path.isdir(full_file_name) and dirs_only) or not dirs_only)) and not filename.startswith('.') and (
                                (
                                    files_filter is not None and
                                    (
                                            (callable(files_filter) and files_filter(full_file_name))
                                            or
                                            (isinstance(files_filter, (list, str, dict)) and filter_files_by_criteria(full_file_name, files_filter))
                                    )
                                )
                                or files_filter is None):
                            accept = True
                            if depth >= 0:
                                items = root.replace(src_root_path, '').split(os.path.sep)
                                if (not explicit_depth and len(items) > depth) or (explicit_depth and len(items) != depth):
                                    accept = False
                            cnt_accepted += 1
                            if cnt_accepted not in file_index_found and len(file_index_found):
                                continue
                            if accept:
                                files.append(os.path.join(root, filename))
                                if len(files) >= file_limit:
                                    raise BreakAllLoops
        except BreakAllLoops:
            pass

    if banned_strings:
        banned_strings = [banned_strings] if not isinstance(banned_strings, list) else banned_strings
        for n in range(len(files)):
            for b in banned_strings:
                if b in files[n]:
                    files[n] = None
                    break
        files = [f for f in files if f is not None]
    return files


def map_paths_by_filename(paths, suffix=None, mode='raw', entry_key_modifier=None, custom_modifier=None, glue='/', keep_one=False):
    """
    Generate a map with keys being names of files or uids and the entries being paths

    TODO:
    generalize with_parent mode to get several levels (one by default)

    :param paths: a path or a list of paths to be mapped
    :param suffix: explicit suffix to remove from filename to get the uid value
    :param mode:    mode of scanning files. Two modes are supported:
                    - raw (by default), setting filename as a key;
                    - custom, needs a custom_modifier callable to apply on the whole path (before entry_key_modifier)
                    - with_parent, gets name of parent dir
                    - uid, setting uid (as instanceUid) as a key;
    :return: the dictionary of paths
    """
    mapped_paths = {}
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        try:
            if custom_modifier is not None and callable(custom_modifier):
                entry_key = custom_modifier(path)
            elif mode == 'uid':
                entry_key = extract_uid_from_path(path, suffix)
            elif mode == 'with_parent':
                filename = os.path.basename(path)
                pardir = os.path.basename(os.path.dirname(path))
                entry_key = f'{pardir}{glue}{filename}'
            else:
                entry_key = os.path.basename(path)
            if entry_key_modifier is not None and callable(entry_key_modifier):
                entry_key = entry_key_modifier(entry_key)
            if entry_key in mapped_paths and not keep_one:
                if not isinstance(mapped_paths[entry_key], list):
                    mapped_paths[entry_key] = [mapped_paths[entry_key]]
                mapped_paths[entry_key].append(path)
            else:
                mapped_paths[entry_key] = path
        except Exception as e:
            print('Failed to map {} because of {}'.format(path, e))

    return mapped_paths
