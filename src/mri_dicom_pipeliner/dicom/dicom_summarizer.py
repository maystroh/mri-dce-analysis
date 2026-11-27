import os
import re
import numpy as np
from mri_dicom_pipeliner.dicom.dicom_walker import DicomWalker
from mri_dicom_pipeliner.dicom.dicom_processing import DicomProcessor
import pandas as pd
import _mri_toolbox.utils.dicom as dcmutils


class DicomSummarizer(DicomWalker):


    series_size_tolerance = 2
    case_id_logic = [['PatientID', 'Modality', ['StudyInstanceUID', 'StudyDate', 'StudyID']]]

    def orientation_from_dicom(self, item):
        try:
            orient = [float(x) for x in item.get('ImageOrientationPatient', 0)]
            row_cosine = np.array(orient[0:3])
            col_cosine = np.array(orient[3:6])
            normal = np.cross(row_cosine, col_cosine)

            # Take absolute values to find dominant axis
            axis = np.argmax(np.abs(normal))

            if axis == 0:
                return "Sagittal"
            elif axis == 1:
                return "Coronal"
            elif axis == 2:
                return "Axial"
            else:
                return "Unknown"
        except Exception as e:
            print(f"Failed getting orientation {e}")
            return "Unknown"

    def classify_weighting(self, item):
        tr = item.get("RepetitionTime", 0)
        te = item.get("EchoTime", 0)
        if tr < 1000 and te < 20:
            return "T1"
        elif tr > 2000 and te > 60:
            return "T2"
        else:
            return "Other/Unknown"

    def generate_series_description(self, dcm, group_key=None):
        data_desc = dcmutils.get_metadata_from_dataset(dcm, normalized_keys='camel_to_snake')
        return data_desc

    def exclude_series(self, series_data, reason):
        if 'details' not in series_data:
            series_data['details'] = []
        series_data['details'].append(reason)
        series_data['exclude'] = True

    def get_case_id(self, dcm):
        case_id_logic = self.get_config('case_id_logic', self.case_id_logic, comment='Defines the logic of grouping series in studies')
        return self.get_grouping_key(dcm, case_id_logic)

    def get_dicom_for_group(self, grouped_entry_elts, mapped_metadata):
        dicom_input = []
        for grouped_entry_elt in grouped_entry_elts:
            # A group can name an entry that carries no DICOM metadata -- TCIA ships a
            # LICENSE file inside every series directory. Skip those instead of passing
            # None into reorder_dicoms, which reads .InstanceNumber off it and crashes.
            dcm = mapped_metadata.get(grouped_entry_elt)
            if dcm is not None:
                dicom_input.append(dcm)
        if not dicom_input:
            return None
        dicom_map = DicomProcessor.reorder_dicoms(dicom_input)
        dcm = list(dicom_map.values())[0]
        return dcm

    def update_series_data(self, series_data, dcm):
        if series_data.get('modality') == 'MR':
            try:
                series_data['weighting'] = self.classify_weighting(dcm)
                series_data['orientation'] = self.orientation_from_dicom(dcm)
            except Exception as e:
                print(f"Failure updating {e}")


    def generate_series_data(self, group_key, grouped_entry_elts, mapped_metadata):
        series_data = {}
        dcm = self.get_dicom_for_group(grouped_entry_elts, mapped_metadata)
        if dcm is None:
            # The group holds no readable DICOM at all (e.g. a TCIA series directory whose
            # only extra file is LICENSE). There is no series to describe.
            return None
        case_id = self.get_case_id(dcm)
        series_data.update(self.generate_series_description(dcm, group_key))
        series_data['case_id'] = case_id
        series_data['size'] = len(grouped_entry_elts)
        self.update_series_data(series_data, dcm)
        return series_data

    def update_cases_map(self, cases_map, group_key, grouped_entries, mapped_metadata):
        for grouped_entry_key, grouped_entry_elts in grouped_entries.items():
            series_data = self.generate_series_data(group_key, grouped_entry_elts, mapped_metadata)
            if series_data is None:
                continue
            case_data = cases_map.setdefault(series_data['case_id'], {})
            case_data[grouped_entry_key] = series_data
            if '_sizes' not in case_data:
                case_data['_sizes'] = {}
            case_data['_sizes'][grouped_entry_key] = case_data[grouped_entry_key]['size']

    def generate_case_map_from_groups(self, mapped_groups):
        cases_map = {}
        mapped_metadata = mapped_groups.get('dicoms', {}).get('metadata')
        groups = mapped_groups.get('groups')
        for group_key, grouped_entries in groups.items():
            self.update_cases_map(cases_map, group_key, grouped_entries, mapped_metadata)
        return cases_map

    def filter_mr_series(self, series_id, series_data, case_data):
        if series_data.get('modality') != 'MR':
            return True
        series_size_grouping = case_data.setdefault('_series_size_grouping', {})
        series_size_tolerance = self.get_config('series_size_tolerance', self.series_size_tolerance,
                                                comment="Tolerance how close could be the series to be grouped together")
        if series_data['size'] < 5:
            self.exclude_series(series_data, 'size')
        if series_data.get('weighting') != 'T1':
            self.exclude_series(series_data, 'weighting')
        if re.search(r"(STIR|MIP|SUB|Survey|_WI|_WO|Disco|test)|(^\(\d+\/)|(T2)", series_data['series_description'],
                     re.IGNORECASE):
            self.exclude_series(series_data, 'description')

        if not series_data.get('exclude'):

            size_group = None
            for series_size_group_key, series_size_group_data in series_size_grouping.items():
                if np.any(np.isclose(series_size_group_data.get('sizes'), series_data.get('size'), rtol=0,
                                     atol=series_size_tolerance)):
                    size_group = series_size_group_key
                    break
            if size_group is None:
                size_group = series_data.get('size')
                series_size_grouping[size_group] = {'sizes': [], 'series': []}
            series_size_grouping[size_group]['sizes'].append(series_data['size'])
            series_size_grouping[size_group]['series'].append(series_id)

    def filter_mg_series(self, series_id, series_data, case_data):
        if series_data.get('modality') != 'MG':
            return

        if 'FOR PROCESSING' in series_data.get('presentation_intent_type', ''):
            self.exclude_series(series_data, 'presentationIntentType')

    def prepare_case_data(self, case_data):
        case_data.setdefault('_series_size_grouping', {})

    def filter_cases(self, case_data, case_id):
        self.prepare_case_data(case_data)
        for series_id, series_data in case_data.items():
            if series_id == '_sizes' or series_id == '_series_size_grouping':
                continue
            series_data['series_id'] = series_id
            series_data['case_id'] = case_id
            self.filter_mr_series(series_id, series_data, case_data)
            self.filter_mg_series(series_id, series_data, case_data)


        return case_data

    def label_mr_series(self, case_data):
        try:
            series_size_grouping = case_data.get('_series_size_grouping', {})
            # Get the biggest group amongtst the grouped series
            sorted_groups = sorted(series_size_grouping.keys())
            if len(sorted_groups) == 0:
                return
            biggest_group = sorted_groups[-1]
            selected_series = series_size_grouping[biggest_group]['series']

            relabeled_series = {}
            dump_series = {}
            for selected_series_key in selected_series:
                series_data = case_data[selected_series_key]
                if selected_series_key not in relabeled_series:
                    if re.search(r'^(?=.*pre)(?!.*post)', series_data['series_description'], re.IGNORECASE):
                        label = 'pre'
                        case_data[selected_series_key]['label'] = label
                        relabeled_series[selected_series_key] = label
                    else:
                        series_key = f"{series_data['acquisition_time']}-{series_data['temporal_position_identifier']}"
                        dump_series[series_key] = selected_series_key

            dumped_series = sorted(dump_series.keys())
            phase_cnt = 0
            dump_with_pre = False
            for dumped_series_item in dumped_series:
                study_key = dump_series[dumped_series_item]
                if len(relabeled_series) == 0:
                    label = 'pre'
                    dump_with_pre = True
                else:
                    phase_cnt += 1
                    label = f'post{phase_cnt}'
                    if phase_cnt == len(dumped_series) or (dump_with_pre and phase_cnt == len(dumped_series) - 1):
                        label = f'{label}-post_last'

                relabeled_series[study_key] = label
                case_data[study_key]['label'] = label

            for series_id, series_data in case_data.items():
                if series_data.get('exclude'):
                    if series_data.get('weighting') == 'T2':
                        series_data['label'] = 't2'

                    elif re.search(r"SUB", series_data['series_description'],
                             re.IGNORECASE):
                        series_data['label'] = 'sub'
        except Exception as e:
            print('Failed labelling MR cases')

    def label_mg_series(self, case_data):
        for series_key, series_data in case_data.items():
            if series_data.get('modality', '') == 'MG':
                series_data['label'] = f"{series_data.get('laterality')}{series_data.get('view')}"

    def label_series(self, case_data):
        self.label_mr_series(case_data)
        self.label_mg_series(case_data)
        print(case_data)
        return case_data

    def store_cases(self, flat_cases):

        df_new = pd.DataFrame(flat_cases).T
        target_path = self.get_cases_store_path()
        if os.path.exists(target_path):
            df_old = pd.read_csv(target_path)
            df_combined = pd.concat([df_new, df_old], ignore_index=True)
            df_target = df_combined.drop_duplicates(subset='series_uid', keep='last')
        else:
            df_target = df_new
        df_target.to_csv(target_path)

        target_path = self.get_config('cases_pickle_target_path', './flat_cases.pickle.gzip',
                                      comment="Storage path for flat cases data in pickle")
        self.update_pickle(path=target_path, data=flat_cases)


    def store_mapping(self, mapped_groups):
        target_path = self.get_config('mapping_target_path', './mapping_store.pickle.gzip',
                                      comment="Storage path for flat cases data")
        self.update_pickle(path=target_path, data=mapped_groups)

    def store_cases_summary(self, cases_summary):
        target_path = self.get_cases_summary_store_path()
        self.update_pickle(path=target_path, data=cases_summary)

    def store_cases_map(self, cases):
        target_path = self.get_cases_map_store_path()
        self.update_pickle(path=target_path, data=cases)

    def store_paths(self, cases):
        target_path = self.get_paths_mapping_store_path()
        self.update_pickle(path=target_path, data=cases)

    def store_data(self, data):

        self.store_cases(data.get('series_data'))
        self.store_mapping(data.get('mapped_groups', {}).get('groups'))
        self.store_paths(data.get('mapped_groups', {}).get('dicoms').get('paths'))
        self.store_cases_summary(data.get('cases_summary'))
        self.store_cases_map(data.get('cases_map'))

    def process_case_data(self, cases_map):
        flat_cases = {}
        cases_summary = {}
        summary_keys = ['case_id', 'study_uid', 'study_description', 'accession_number', 'patient_id', 'patient_birthdate', 'patient_age', 'patient_name', 'study_date', 'study_time', 'modality',]
        for case_id, case_data in cases_map.items():
            case_summary = cases_summary.setdefault(case_id, {})

            for series_id, series_data in case_data.items():
                case_summary.update(dict((key, series_data[key])
                                   for key in summary_keys if key in series_data))

                if series_id == '_sizes' or series_id == '_series_size_grouping':
                    continue
                data = {}
                data.update(series_data)
                flat_cases[series_id] = data

        self.share['cases_map'] = cases_map
        self.share['cases_summary'] = cases_summary
        self.share['series_data'] = flat_cases

        self.store_data(self.share)

    def walk_groups(self, mapped_groups):
        self.share['mapped_groups'] = mapped_groups
        cases_map = self.generate_case_map_from_groups(mapped_groups)

        for case_id, case_data in cases_map.items():

            case_data = self.filter_cases(case_data, case_id)
            self.label_series(case_data)

        self.process_case_data(cases_map)
