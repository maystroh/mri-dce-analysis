from _mri_toolbox.utils.logging import print
from _mri_toolbox.framework.backbone import Backbone
import _mri_toolbox.utils.dicom as dcmutils


class DicomCommons(Backbone):

    def extract_data_for_logic(self, dicom, logic_groupings, get_first=True):
        """
        For a given dicom Dataset object, combine data to be used as logical grouping

        By default takes the first acceptable amongst the listed logics

        The groupings are the lists of strings or lists as follows

        ```
        [["SeriesInstanceUID", ["Laterality", "ImageLaterality"], ["extra:view"]]]
        ```

        Each string value shall be availabe, for the lists, at least one shall be available and the first will be kept.

        :param dicom: Dataset object
        :param logic_groupings: List of groupings
        :param get_first: use the first acceptable (all required fields have values)
        :return:
        """
        groupings = []
        for logic_entry in logic_groupings:
            grouping = []
            for item in logic_entry:
                val = dcmutils.extract_first_value_from_dicom(dicom, item)
                if val:
                    grouping.append(val)
            if len(grouping) == len(logic_entry):
                groupings.append(grouping)
                if get_first:
                    return grouping

    def get_grouping_key(self, ds, logic):
        return tuple(self.extract_data_for_logic(ds, logic))