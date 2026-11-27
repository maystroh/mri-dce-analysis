from copy import deepcopy
import numpy as np
import dicom2nifti
import dicom2nifti.common as dcm2niicommon

class DicomProcessor():

    def __init__(self, config={}):
        self.config = (config or {})
        if config.get('disable_validate_slice_increment'):
            dicom2nifti.settings.disable_validate_slice_increment()

    @staticmethod
    def reorder_dicoms(dicom_input, key_type='InstanceNumber:int'):
        dicom_map = dict()
        for i in range(len(dicom_input)):
            if key_type == 'InstanceNumber:int':
                key = int(dicom_input[i].InstanceNumber)
            dicom_map[key] = dicom_input[i]

        return {k: v for k, v in sorted(dicom_map.items())}

    def guarantee_completeness(self, dicom_input):
        dicom_map = DicomProcessor.reorder_dicoms(dicom_input)
        if sorted(list(dicom_map.keys()))[-1] > len(dicom_map):
            updated_map = []
            sorted_keys = sorted(list(dicom_map.keys()))
            diffs = []

            prev = None
            for key in sorted_keys:
                if prev is not None:
                    diffs.append(key - prev)
                prev = key
            for idx in sorted_keys:
                item = dicom_map[idx]
                if (idx-prev) > np.mean(diffs):
                    prev_item = dicom_map[prev]
                    modified_item = deepcopy(item)
                    modified_item.ImagePositionPatient = list((np.array(modified_item.ImagePositionPatient)+np.array(prev_item.ImagePositionPatient))/2)
                    modified_item.InstanceNumber = str(idx-1)
                    img = ((modified_item.pixel_array+prev_item.pixel_array)/2).astype(np.uint16)
                    modified_item.PixelData = img.tobytes()
                    updated_map.append(modified_item)
                updated_map.append(item)
                prev = idx
            return updated_map
        return list(dicom_map.values())

    def is_vaild_dicom(self, ds):
        return dicom2nifti.common.is_valid_imaging_dicom(ds)
    def generate_nifti(self, dicom_input, file_path, guarantee_completeness=True, reorient_nifti=False):
        """Convert one series to NIfTI."""
        if guarantee_completeness:
            dicom_input = self.guarantee_completeness(dicom_input)
        return dicom2nifti.convert_dicom.dicom_array_to_nifti(dicom_input, file_path, reorient_nifti=reorient_nifti)