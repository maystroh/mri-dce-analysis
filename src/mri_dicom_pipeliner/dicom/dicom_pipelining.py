from mri_dicom_pipeliner.dicom.dicom_common_walker import DicomCommonWalker
from mri_dicom_pipeliner.dicom.dicom_commons import DicomCommons
from mri_dicom_pipeliner.dicom.dicom_grouper import DicomGrouper
from mri_dicom_pipeliner.dicom.dicom_summarizer import DicomSummarizer
from _mri_toolbox.framework.configurable import Configurable
from _mri_toolbox.utils.logging import print
from _mri_toolbox.framework.argumentable import Argumentable
import _mri_toolbox.utils.dicom as dcmutils

class DicomPipelining(Argumentable, DicomCommons):
    """
    DicomPipelining gets dicoms from a given source (e.g., storage_path).
    """

    logic_map = {
        'MG_default': {"filter": [{'tag': 'Modality', 'condition': 'equals', 'value': 'MG'}],
                       "groupings": [["SeriesInstanceUID", ["Laterality", "ImageLaterality"], ["extra:view"]]]},
        'US_default': {"filter": [{'tag': 'Modality', 'condition': 'equals', 'value': 'US'}],
                       "groupings": [["SeriesInstanceUID"]]},
        'MR_default': {"filter": [{'tag': 'Modality', 'condition': 'equals', 'value': 'MR'}],
                       "groupings": [["PatientID", "StudyDate", "SeriesInstanceUID", "TemporalPositionIdentifier", ],
                                     ["PatientID", "StudyDate", "SeriesInstanceUID", "AcquisitionNumber", ],
                                     ["PatientID", "StudyDate", "SeriesInstanceUID"]]},
    }

    def get_groups_ids_for_dicom(self, ds, logic_map):
        """
        Get all logical group values for a given dicom file
        :param ds:
        :param logic_map:
        :return:
        """
        group_ids = {}
        for logic_code, logic_item in logic_map.items():
            if dcmutils.filter_dicom_entry(ds, logic_item.get('filter', {})):
                group_ids.setdefault(logic_code, set())
                group_ids[logic_code].add(self.get_grouping_key(ds, logic_item.get('groupings', {})))
        return group_ids

    def dicom_logical_group(self, dicoms, logic_map):
        """
        Group listed dicoms using logic_map config
        :param dicoms:
        :param logic_map:
        :return:
        """
        unique_groups = {}
        dicoms_metadata = dicoms.get('metadata', dicoms)
        for uid, ds in dicoms_metadata.items():
            dicom_groups_ids = self.get_groups_ids_for_dicom(ds, logic_map)
            for group_key, item in dicom_groups_ids.items():
                unique_groups.setdefault(group_key, {})
                for it in item:
                    unique_groups[group_key].setdefault(it, set())
                    unique_groups[group_key][it].add(ds.get('SOPInstanceUID'))
        return {'dicoms': dicoms, 'groups': unique_groups}

    def read_dicom(self, path):
        """
        Wrap read_dicom from path, handling the exception
        :param path:
        :return:
        """
        try:
            ds = dcmutils.read_dicom(path, stop_before_pixels=True)
            return ds
        except Exception as e:
            print(f'Failure reading file as dicom {e}')
        return None

    def collect_mapped_dicoms(self):
        """
        Map dicom metadata and map them to uids
        :return:
        """
        # Scan with `with_parent` so files that share a name across series directories do
        # not collide into a list (TCIA numbers every series 00000001.dcm), then RE-KEY the
        # map on the SOPInstanceUID read from each file's CONTENT.
        #
        # This matters because the logical groups identify their members by real
        # SOPInstanceUID. The historical `mode='uid'` derived that UID from the FILENAME,
        # which only works when files are named <SOPInstanceUID>.dcm -- this pipeline's own
        # export convention. On any other naming the group lookup missed every entry and the
        # case map came out empty. Reading the UID from the file is convention-independent,
        # and identical to the old key whenever the old assumption held.
        scan_mode = self.get_config(
            'dicom_path_map_mode', 'with_parent',
            comment="How files are scanned before being re-keyed by SOPInstanceUID: "
                    "'with_parent' (default, tolerates repeated filenames) or 'uid'")
        scanned = self.get_mapped_paths(
            self.get_config('src_path', comment="Dicom source path"), mode=scan_mode)
        paths = {}
        dicoms = {}
        for _, p in scanned.items():
            if isinstance(p, list):          # still colliding: keep every one of them
                candidates = p
            else:
                candidates = [p]
            for candidate in candidates:
                ds = self.read_dicom(candidate)
                if not ds:
                    continue
                uid = str(getattr(ds, 'SOPInstanceUID', '') or '')
                if not uid:
                    continue
                paths[uid] = candidate
                dicoms[uid] = ds
        return {'paths': paths, 'metadata': dicoms}


    def run_walkers(self, logical_groups):
        self.share = {}
        walkers = self.get_config('walkers', [], comment="List of the logical items walkers")
        for walker_item in walkers:

            if walker_item == 'dicom2nifti':
                walker = DicomCommonWalker(parent=self)
            elif walker_item == 'cases_mapper':
                walker = DicomSummarizer(parent=self)
            elif walker_item == 'cases_grouper':
                walker = DicomGrouper(parent=self)
            walker.inject_share(self.share)
            walker.walk_groups(logical_groups)
            self.share.update(walker.share)


    @Configurable.autodoc('file', True)
    def process(self):
        logic_map = self.get_config('logic_map', self.logic_map, comment='Dicom logical mapping')
        dicoms = self.collect_mapped_dicoms()
        logical_groups = self.dicom_logical_group(dicoms, logic_map)
        self.run_walkers(logical_groups)
