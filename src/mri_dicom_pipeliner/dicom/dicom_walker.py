from abc import abstractmethod
import os
from mri_dicom_pipeliner.dicom.dicom_commons import DicomCommons
from _mri_toolbox.utils.logging import print
from _mri_toolbox.framework.backbone import Backbone
import re

class DicomWalker(DicomCommons, Backbone):

    cases_summary_store_path = './cases_summary.pickle.gzip'
    cases_store_path = './flat_cases_store.csv'
    cases_map_store_path = './mapped_cases_store.pickle.gzip'
    merged_series_map_store_path = './merged_series_store.pickle.gzip'
    paths_mapping_store_path = './paths_mapping.pickle.gzip'

    def __init__(self, config=None, **kwargs):
        super(DicomWalker, self).__init__(config, **kwargs)
        self.share = {}

    def inject_share(self, share, destruct=True):
        self.share = share
        if destruct:
            for key, value in share.items():
                if not hasattr(self, key):
                    setattr(self, key, value)

    def get_cases_summary_store_path(self):
        return self.get_config('cases_summary_store_path', self.cases_summary_store_path,
                               comment="Storage path for flat cases data")

    def get_cases_map_store_path(self):
        return self.get_config('cases_map_store_path', self.cases_map_store_path,
                               comment="Storage path for mapped cases data")

    def get_paths_mapping_store_path(self):
        return self.get_config('paths_mapping_store_path', self.paths_mapping_store_path,
                               comment="Storage path for mapped paths to instances")

    def get_merged_series_map_store_path(self):
        return self.get_config('merged_series_map_store_path', self.merged_series_map_store_path,
                               comment="Storage path for mapped merged series data")

    def get_cases_store_path(self):
        return self.get_config('cases_store_path', self.cases_store_path,
                               comment="Storage path for flat cases data")

    def update_pickle(self, data={}, path=None):

        if os.path.exists(path):
            data_store = self.read_pickle(path)
            self.update_nested_dict(data_store, data)
        else:
            data_store = data

        self.dump_pickle(filename=path, data=data_store, timestamped=False)

    @abstractmethod
    def walk_groups(self, logical_groups):
        pass

    def check_filter(self, val, filter_item, accept_default=True):
        accept = accept_default
        if filter_item:
            if 'in' in filter_item:
                accept = False
                label_in_filter = self.make_list(filter_item['in'])
                try:
                    accept = val in label_in_filter
                except Exception as e:
                    print(e)
            if 'regex' in filter_item:
                label_in_filter = filter_item['regex']
                accept = re.search(label_in_filter, val)
        return accept
