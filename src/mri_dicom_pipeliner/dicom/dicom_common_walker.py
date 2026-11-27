from mri_dicom_pipeliner.dicom.dicom_processing import DicomProcessor
from mri_dicom_pipeliner.dicom.dicom_walker import DicomWalker
from mri_dicom_pipeliner.dicom.metadata_exporter import summarize_series, write_csv
import _mri_toolbox.utils.dicom as dcmutils
import os
import re

class DicomCommonWalker(DicomWalker):

    mapped_processings = {"MR_default": ["generate_nifti"]}

    def post_process_label(self, dcm_label, group_key):
        if 'series_label' in dcm_label:
            dcm_label = dcm_label.replace('series_label', self.series_data.get(group_key, {}).get('label'))
        return dcm_label
    def generate_label_for_series(self, dicom_input, group_key):
        """
        Generate a label for a grouped series using provided dicom_input data
        :param dicom_input:
        :param group_key:
        :return:
        """
        nifti_file_path_format = self.get_config('nifti_file_name_format', 'PatientID/StudyDate/{SeriesInstanceUID}-{ContentDate}-{ContentTime:int}-{SeriesDescription:clean}-{const:series_label}', comment="The dicom-values-aware file path for the generated nifti files")
        dicom_map = DicomProcessor.reorder_dicoms(dicom_input)
        first_item = dicom_map[sorted(dicom_map.keys())[0]]
        dcm_label = dcmutils.get_dicom_value_for_strategy(first_item, nifti_file_path_format)
        dest_path = self.get_config('dest_path', default='./', comment="Nifti storage destination path")
        dcm_label = self.post_process_label(dcm_label, group_key)
        file_path = f'{dest_path}/{dcm_label}.nii.gz'

        return file_path

    def generate_nifti(self, mapped_paths, group, group_key):
        """
        Generate a nifti using a map of uids to paths, group of uids and its group_key

        :param mapped_paths:
        :param group:
        :param group_key:
        :return:
        """
        overwrite_nifti = self.get_config('overwrite_nifti',  False)
        processor_config = self.get_config('dicom2nifti_cfg', {'guarantee_completeness': True}, comment='Settings triggers for dicom2nifti. see dicom2nifti.settings for details +  custom guarantee_completeness option filling in missing slices')
        dcm_proc = DicomProcessor(processor_config)
        dicoms = self.clean_dict(mapped_paths, group)
        dicom_paths = list(dicoms.values())
        dicom_input = dcmutils.read_dicom_set(dicom_paths, is_valid_dicom=dcm_proc.is_vaild_dicom)
        file_path = self.generate_label_for_series(dicom_input, group_key)
        conversion_status = 'converted'
        conversion_error = ''
        if os.path.exists(file_path) and not overwrite_nifti:
            conversion_status = 'already_exists'
        else:
            self.get_dir_for_file(file_path)
            try:
                dcm_proc.generate_nifti(
                    dicom_input,
                    file_path,
                    processor_config.get('guarantee_completeness', True),
                    processor_config.get('reorient_nifti', False),
                )
            except Exception as e:
                conversion_status = 'failed'
                conversion_error = str(e)
                print(f'Failed generating nifti for group {group_key} with {e}')

        if self.get_config(
            'export_metadata_csv', True,
            comment='Write one metadata row for every MR series sent to NIfTI conversion',
        ):
            return summarize_series(
                dicom_input,
                dicom_paths,
                nifti_path=file_path,
                conversion_status=conversion_status,
                conversion_error=conversion_error,
            )



    def filter_group_for_processing(self, group_key, processing):
        series_filters = self.get_config('series_filters', {}, comment="Filtering series per processing, _common key stands for all processings")
        series_filter = series_filters.get(processing, series_filters.get('_common', None))
        accept = True
        if series_filter:
            val = self.series_data.get(group_key, {}).get('label', '').lower()
            accept = self.check_filter(val, series_filter.get('label'))


        return accept

    def process_group(self, mapped_paths, group_key, group, processing):
        """
        Process group provided processing to use
        :param mapped_paths:
        :param group_key:
        :param group:
        :param processing:
        :return:
        """
        accept = self.filter_group_for_processing(group_key, processing)
        if not accept:
            return
        if processing == 'generate_nifti':
            metadata = self.generate_nifti(mapped_paths, group, group_key)
            if metadata:
                self.metadata_rows.append(metadata)

    def process_groups(self, mapped_paths,  group_key, groups, processings):
        """
        Process all groups with all of the configured processings

        :param mapped_paths:
        :param group_key:
        :param groups:
        :param processings:
        :return:
        """

        for group_key, group in groups.items():
            for processing in processings:
                self.process_group(mapped_paths, group_key, group, processing)

    def get_processings(self, group_key):
        """
        Collect processings for given logical grouping
        :param group_key:
        :return:
        """
        mapped_processings = self.get_config('mapped_processings', self.mapped_processings, comment="Mappings from logical groups to specific business")
        return mapped_processings.get(group_key, [])

    def walk_groups(self, mapped_groups):
        """
        Given groups, get required processings and process them accordignly
        :param mapped_groups:
        :return:
        """

        self.metadata_rows = []
        mapped_paths = mapped_groups.get('dicoms', {}).get('paths')
        groups = mapped_groups.get('groups')

        for group_key, grouped_entries in groups.items():
            processings = self.get_processings(group_key)
            self.process_groups(mapped_paths, group_key, grouped_entries, processings)

        if self.metadata_rows:
            dest_path = self.get_config('dest_path', default='./')
            metadata_path = self.get_config(
                'metadata_csv_path',
                os.path.join(dest_path, 'dicom_series_info.csv'),
                comment='Destination of the DICOM-to-NIfTI series metadata CSV',
            )
            write_csv(self.metadata_rows, metadata_path)
