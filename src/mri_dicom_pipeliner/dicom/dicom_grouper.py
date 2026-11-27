from mri_dicom_pipeliner.dicom.dicom_walker import DicomWalker
import datetime
import pandas as pd
import _mri_toolbox.utils.dicom as dcmutils
import re

class DicomGrouper(DicomWalker):

    case_dates_match = {'mg:mr':  [60, 20], 'mr:us': [20,60], 'mg:us': [15,15], }
    patient_data_match = ['patient_id', 'patient_birthdate', 'patient_name']

    def compare_older_newer_studies(self, ref_series, candy_series, dates_match):
        ref_date = ref_series.get('study_date', '')
        candy_date = candy_series.get('study_date', '')
        if ref_date and candy_date:
            format_str = "%Y%m%d"
            date1 = datetime.datetime.strptime(str(int(ref_date)), format_str)
            date2 = datetime.datetime.strptime(str(int(candy_date)), format_str)
            diff1 = (date2 - date1).days
            diff2 = (date1 - date2).days
            if diff1 <= dates_match[0] and diff2 <= dates_match[1]:
                return 1
            else:
                return 0

    def compare_study_dates(self, ref_series, candy_series):
        case_dates_match = self.get_config('case_dates_match', self.case_dates_match, comment="")

        ref_modality = ref_series.get('modality', '').lower()
        candy_modality = candy_series.get('modality', '').lower()
        ref_candy_key = f'{ref_modality}:{candy_modality}'
        candy_ref_key = f'{candy_modality}:{ref_modality}'
        case_dates_matcher = None
        if ref_candy_key in case_dates_match:
            case_dates_matcher = ref_candy_key
            a_series = ref_series
            b_series = candy_series

        elif candy_ref_key in case_dates_match:
            case_dates_matcher = candy_ref_key
            a_series = candy_series
            b_series = ref_series
        if case_dates_matcher:
            return self.compare_older_newer_studies(a_series, b_series, case_dates_match[case_dates_matcher])

        return 0

    def normalize_patient_name(self, patient_name):
        return (patient_name or '').replace(' ', '').replace('-', '').lower()

    def compare_patient_names(self, ref_patient_name, candy_patient_name):
        ref_patient_name = self.normalize_patient_name(ref_patient_name)
        candy_patient_name = self.normalize_patient_name(candy_patient_name)
        return candy_patient_name == ref_patient_name and ref_patient_name != ''

    def compare_accession_numbers(self, ref_series, candy_series):
        return self.compare_non_empty_value(ref_series.get('accession_number', 'ref'), candy_series.get('accession_number', 'candy'))

    def compare_non_empty_value(self, ref, candy, ne=0):
        return 0 if ref == '' and candy == '' else 1 if ref == candy else ne

    def compare_patient_data(self, ref_series, candy_series):
        accepts = []
        patient_data_match = self.get_config('patient_data_match', self.patient_data_match, comment="Matching strategies for patient data (ID, name, and birthdate)")
        patient_data_match = '+'.join(self.make_list(patient_data_match))
        patient_id = 0
        if 'patient_id' in patient_data_match:
            patient_id = self.compare_non_empty_value(ref_series.get('patient_id', 'ref'), candy_series.get('patient_id', 'candy'), -1)
            accepts.append(patient_id)

        # Patient inequality if done leads to exclusion of the sample
        # Otherwise check patient id and patient name
        if patient_id >= 0:
            if 'patient_birthdate' in patient_data_match:
                accepts.append(self.compare_non_empty_value(ref_series.get('patient_birthdate', 'ref'), candy_series.get('patient_birthdate', 'candy')))

            if 'patient_name' in patient_data_match:
                ref_patient_name = ref_series.get('patient_name', 'ref')
                candy_patient_name = candy_series.get('patient_name', 'candy')
                accepts.append(1 if self.compare_patient_names(ref_patient_name, candy_patient_name) else 0)

        return accepts

    def try_cases_merge(self, ref_series, candy_series):
        accepts = []
        if 1 in (self.make_list(self.compare_patient_data(ref_series, candy_series))):
            accepts.extend(self.make_list(self.compare_accession_numbers(ref_series, candy_series)))
            accepts.extend(self.make_list(self.compare_study_dates(ref_series, candy_series)))

        # -1 stands for merge refusal. 1 stands for merge accept. 0 stands for no-opinion
        return -1 not in accepts and 1 in accepts

    def generate_virtual_study_id(self, series_items):
        return dcmutils.generate_uid(None, 'study_uid', persistent_uids=False)

    def merge_studies(self, series_data):
        series_items = series_data
        if isinstance(series_data, dict):
            series_items = list(series_data.values())

        for i in range(len(series_items)):
            ref_series = series_items[i]
            for j in range(len(series_items)):
                candy_series = series_items[j]
                if i == j:
                    continue
                if self.try_cases_merge(ref_series, candy_series) > 0:
                    ref_series.setdefault('merge', []).append(candy_series['case_id'])
                    merge_id = ref_series.get('merge_id', candy_series.get('merge_id', self.generate_virtual_study_id([ref_series, candy_series])))
                    ref_series['merge_id'] = merge_id
                    candy_series['merge_id'] = merge_id

        return series_data

    def check_series_config(self, series_item, combine_config):
        # for combine_setting in combine_config:
        field = combine_config.get('field')
        rule = combine_config.get('rule')
        return self.check_filter(series_item.get(field, '').lower(), rule)


    def check_combo(self, combined_series, combined_conditions):
        gte = combined_conditions.get('gte')
        accepts = []
        if gte and combined_series and len([x for x in combined_series.values() if x]) >= gte:
            accepts.append(True)
        return True in accepts and False not in accepts

    def try_series_combiner(self, case_data, combine_config={}):
        combined_series = {}
        combine_items = combine_config.get('items', {})
        combine_conditions = combine_config.get('conditions', {})

        for combine_id, combine_config in combine_items.items():
            combined_series[combine_id] = None
            for series_id, series_item in case_data.items():
                if self.check_series_config(series_item, combine_config):
                    combined_series[combine_id] = series_id

        if self.check_combo(combined_series, combine_conditions):
            return combined_series
        else:
            return None

    def merge_series(self, cases_data):
        series_combiners = self.get_config('series_combiners', [
            {'id': 'pre-post1-postN', 'items': {0: {'field': 'label', 'rule': {'in': ['pre']}}, 1: {'field': 'label', 'rule': {'regex': 'post1'}}, 2: {'field': 'label', 'rule': {'regex': 'post_last'}}}, 'conditions': {'gte': 2}},
            {'id': 'post1', 'items': {0: {'field': 'label', 'rule': {'regex': 'post1'}}}, 'conditions': {'gte': 1}}])
        series_combiners = self.make_list(series_combiners)
        combined = {}
        for case_id, case_data in cases_data.items():
            for series_combiner in series_combiners:
                combined_series = self.try_series_combiner(case_data, series_combiner)
                if combined_series:
                    combined.setdefault(case_id, {})
                    combined[case_id][series_combiner.get('id')] = combined_series

        return combined
    def load_case_summary_data(self):
        target_path = self.get_cases_summary_store_path()
        data = self.read_pickle(target_path)
        return data

    def load_cases_data(self):
        target_path = self.get_cases_map_store_path()
        data = self.read_pickle(target_path)
        return data

    def update_reference_ids(self, data):

        for item_id, item in data.items():
            item['reference_id'] = item.get('merge_id', item.get('study_uid'))
        return data

    def group_cases(self):
        data = self.load_case_summary_data()
        data = self.merge_studies(data)
        data = self.update_reference_ids(data)

        self.update_pickle(path=self.get_cases_summary_store_path(), data=data)
        pd.DataFrame(data).T.to_csv('./merged_studies.csv')

    def group_series(self):
        data = self.load_cases_data()
        data = self.merge_series(data)
        self.update_pickle(path=self.get_merged_series_map_store_path(), data=data)

    def process(self):
        self.group_cases()
        self.group_series()
