import hashlib
import numpy as np
import pydicom


import _mri_toolbox.utils.data as datautils
import datetime
from pydicom import DataElement

from _mri_toolbox.utils.data import CaseInsensitiveDict

uid_root = '2.16.250.828871699.1.999.'

view_code_samples = {}
view_code_samples[
    'cranio-caudal'] = "{'Code Value': 'R-10242', 'Coding Scheme Designator': 'SNM3', 'Code Meaning': 'cranio-caudal'}"

view_code_samples[
    'latero-medial'] = "{'Code Value': 'R-10228', 'Coding Scheme Designator': 'SNM3', 'Code Meaning': 'latero-medial'}"

view_code_samples[
    'medio-lateral'] = "{'Code Value': 'R-10224', 'Coding Scheme Designator': 'SNM3', 'Code Meaning': 'medio-lateral'}"
view_code_samples[
    'medio-lateral oblique'] = "{'Code Value': 'R-10226', 'Coding Scheme Designator': 'SNM3', 'Code Meaning': 'medio-lateral oblique'}"

view_code_samples['CC'] = view_code_samples['cranio-caudal']
view_code_samples['MLO'] = view_code_samples['medio-lateral oblique']
view_code_samples['LM'] = view_code_samples['latero-medial']
view_code_samples['LM'] = view_code_samples['medio-lateral']

view_code_modifiers_samples = {}
view_code_modifiers_samples['Cleavage'] = {'scheme': 'SRT', 'code': 'R-102D2', 'meaning': 'Cleavage'}
view_code_modifiers_samples['Axillary Tail'] = {'scheme': 'SRT', 'code': 'R-102D1', 'meaning': 'Axillary Tail'}
view_code_modifiers_samples['Rolled Lateral'] = {'scheme': 'SRT', 'code': 'R-102D3',
                                                 'meaning': 'Rolled Lateral'}
view_code_modifiers_samples['Rolled Medial'] = {'scheme': 'SRT', 'code': 'R-102D4', 'meaning': 'Rolled Medial'}
view_code_modifiers_samples['Rolled Inferior'] = {'scheme': 'SRT', 'code': 'R-102CA',
                                                  'meaning': 'Rolled Inferior'}
view_code_modifiers_samples['Rolled Superior'] = {'scheme': 'SRT', 'code': 'R-102C9',
                                                  'meaning': 'Rolled Superior'}
view_code_modifiers_samples['Implant Displaced'] = {'scheme': 'SRT', 'code': 'R-102D5',
                                                    'meaning': 'Implant Displaced'}
view_code_modifiers_samples['Magnification'] = {'scheme': 'SRT', 'code': 'R-102D6', 'meaning': 'Magnification'}
view_code_modifiers_samples['Spot Compression'] = {'scheme': 'SRT', 'code': 'R-102D7',
                                                   'meaning': 'Spot Compression'}
view_code_modifiers_samples['Tangential'] = {'scheme': 'SRT', 'code': 'R-102C2', 'meaning': 'Tangential'}
view_code_modifiers_samples['Nipple in profile'] = {'scheme': 'SRT', 'code': 'R-40AB3',
                                                    'meaning': 'Nipple in profile'}
view_code_modifiers_samples['Anterior compression'] = {'scheme': 'SRT', 'code': 'P2-00161',
                                                       'meaning': 'Anterior compression'}
view_code_modifiers_samples['Infra-mammary fold'] = {'scheme': 'SRT', 'code': 'R-40ABE',
                                                     'meaning': 'Infra-mammary fold'}
view_code_modifiers_samples['Axillary tissue'] = {'scheme': 'SRT', 'code': 'R-40AB2',
                                                  'meaning': 'Axillary tissue'}
view_code_modifiers_samples['Axillary tissue'] = {'scheme': 'SCT', 'code': 'R-40AB2',
                                                  'meaning': 'Implant Displaced'}


def read_dicom_set(dicoms, stop_before_pixels=False, to_map=False, is_valid_dicom=None, order_key='SOPInstanceUID', to_sort=True, output='dicom'):
    """
    Read all dicom files in a given directory (stop before pixels)


    :param stop_before_pixels: Should we stop reading before the pixeldata (handy if we only want header info)
    :param to_map: Make it dict or list,
    :param is_valid_dicom: function to verify whether it is a valid dicom,
    :param order_key: use order key
    :param to_sort: sort the list by keys
    :param output: output dicom Datasets or just paths

    :return: List of dicom objects
    """
    dicom_input = {}
    for dicom_file in dicoms:
        try:
            dicom_headers = pydicom.read_file(dicom_file,
                                              defer_size="1 KB",
                                              stop_before_pixels=stop_before_pixels,
                                              force=False)
            if not is_valid_dicom or (is_valid_dicom and callable(is_valid_dicom) and is_valid_dicom(dicom_headers)):
                if isinstance(order_key, str):
                    order_value = dicom_headers.get(order_key)
                elif callable(order_key):
                    order_value = order_key(dicom_headers)
                if output == 'paths':
                    dicom_input[order_value] = dicom_file
                else:
                    dicom_input[order_value] = dicom_headers
        except Exception as e:
            print(f'Failed reading {dicom_file} with {e}')
    if to_map:
        return dicom_input
    else:
        if to_sort:
            return list([v for k, v in sorted(dicom_input.items())])
        else:
            return list(dicom_input.values())

def read_dicom(*args, **kwargs) -> pydicom.Dataset :
    return pydicom.read_file(*args, **kwargs)


def get_view(ds_item):
    view = 'mlo' if dicom_is_view_mlo(ds_item) else 'cc' if dicom_is_view_cc(ds_item) else 'ml'
    return view.upper()


def get_laterality(ds_item, wordy=False):
    laterality = (ds_item.Laterality if hasattr(ds_item,
                                           'Laterality') and ds_item.Laterality is not None and ds_item.Laterality != '' else ds_item.ImageLaterality if hasattr(
        ds_item, 'ImageLaterality') and ds_item.ImageLaterality is not None else 'na').upper()
    if laterality is not None and wordy:
        if laterality.lower() == 'r':
            laterality = 'right'
        if laterality.lower() == 'l':
            laterality = 'left'
    return laterality

def get_imaging_type(ds_item):
    return ds_item.Modality

def get_patient_name(ds_item):
    return str(ds_item.PatientName)

def get_pixel_spacing(ds_item):
    try:
        return [ds_item.PixelSpacing[0], ds_item.PixelSpacing[1]]
    except:
        try:
            return [ds_item.ImagerPixelSpacing[0], ds_item.ImagerPixelSpacing[1]]
        except:
            return []


def enrich_metadata_config(config={}):
    config = (config or {})
    if 'dicom_fields_map' not in config:
        config['dicom_fields_map'] = {"uid": 'SOPInstanceUID', "imageType": 'ImageType', "studyUid": 'StudyInstanceUID',
             "seriesUid": 'SeriesInstanceUID', "studyDescription": "StudyDescription",
             "seriesDescription": "SeriesDescription",
             "accessionNumber": 'AccessionNumber', "patientId": 'PatientID',
             "patientBirthdate": "PatientBirthdate", "instanceNumber": "InstanceNumber", "studyDate": "StudyDate",
             "studyTime": "StudyTime", "seriesDate": "SeriesTime",
             "seriesTime": "SeriesTime", "patientAge": 'PatientAge',
             "patientName": get_patient_name, "modality": 'Modality',
             "imaging_type": get_imaging_type,
             "width": "Columns", "height": "Rows", "view": get_view,
             "manufacturer": "Manufacturer",
             "pixelSpacing": get_pixel_spacing, 'uid': 'SOPInstanceUID'}

    if 'MG' not in config:
        config['MG'] = {
            "compression": "CompressionForce", "orientation": get_laterality, "laterality": get_laterality,
            "presentationIntentType": "PresentationIntentType",
            "thickness": "BodyPartThickness", "tubeCurrent": "XRayTubeCurrent", "organDose": "OrganDose",
            "exposure": "Exposure", "exposureTime": "ExposureTime", "voltage": "KVP",
            "primaryAngle": "PositionerPrimaryAngle"
        }

    if 'MR' not in config:
        config['MR'] = {"TemporalPositionIdentifier": "TemporalPositionIdentifier",
                        "ScanningSequence": 'ScanningSequence',
                        "ImageOrientationPatient": 'ImageOrientationPatient',
                        "SequenceVariant":"SequenceVariant", "ContrastBolusAgent":"ContrastBolusAgent", "ContrastBolusVolume": "ContrastBolusVolume","ContrastBolusTotalDose": "ContrastBolusTotalDose", "AcquisitionNumber": "AcquisitionNumber",  "ImagesInAcquisition": "ImagesInAcquisition", "SeriesNumber": "SeriesNumber", "RepetitionTime": "RepetitionTime", "EchoTime": "EchoTime", "AcquisitionTime": "AcquisitionTime","ContentTime": "ContentTime",  "SequenceName": "SequenceName", "NumberOfTemporalPositions": "NumberOfTemporalPositions"}
    return config


def get_metadata_from_dataset(ds, config={}, case_insensitive=False, normalized_keys=None):

    config = enrich_metadata_config(config)

    if ds is None:
        return {}

    dicom_fields_map = config.get('dicom_fields_map')
    dicom_fields_map_addons = config.get('dicom_fields_map_addons', {})
    dicom_fields_map.update(dicom_fields_map_addons)
    dicom_fields_map.update({"modality": "Modality"})
    item_data = {}

    def update_item_data_with_params(dicom_fields):
        for field_name, field in dicom_fields.items():
            try:
                if callable(field):
                    item_data[field_name] = field(ds)
                else:
                    item_data[field_name] = ds.get(field)
            except:
                pass

    update_item_data_with_params(dicom_fields_map)
    modality = item_data.get('modality')

    update_item_data_with_params(config.get(modality, {}))

    if case_insensitive:
        return CaseInsensitiveDict(item_data)
    elif normalized_keys:
        return datautils.normalize_keys(item_data, norm=normalized_keys)

    return item_data

def filter_dicom_entry(ds, conditions=None):
    if ds is None:
        return False
    if conditions == None or len(conditions) == 0:
        return True
    accepts = []
    if conditions is not None and len(conditions) > 0:
        if isinstance(conditions, dict):
            accept = True
            if 'reject' in conditions:
                for item in conditions['reject']:
                    if item in ds:
                        strategy = conditions['reject'][item].get('strategy', 'one')
                        if strategy == 'one':
                            if ds[item].value in conditions['reject'][item].get('values', []):
                                accept = False

            return accept

        elif isinstance(conditions, list):
            for condition in conditions:
                accept_null = condition.get('accept_null', False)
                tag = condition.get('tag', None)
                cond = condition.get('condition', None)
                decorators = condition.get('decorators', [])

                value = condition.get('value', None)
                smart_value = condition.get('smart_value', None)
                if smart_value is not None:
                    if smart_value == 'today':
                        value = datetime.datetime.now().strftime("%Y%m%d")
                if tag is None or cond is None:
                    continue
                val = ds.get(tag, None)

                if val is None:
                    if cond == 'is_none' or  cond == 'is_null':
                        accepts.append(True)
                        continue
                    elif not accept_null:
                        accepts.append(False)
                        continue
                    else:
                        continue
                if val is not None:
                    if isinstance(val, DataElement):
                        val = val.value
                if decorators is not None and len(decorators) > 0:
                    if 'lower' in decorators:
                        try:
                            val = val.lower()
                        except:
                            pass
                    if 'upper' in decorators:
                        try:
                            val = val.upper()
                        except:
                            pass
                if cond == 'is_none' or  cond == 'is_null':
                    accepts.append(False)
                elif cond == 'equals':
                    if val == value:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'not_equals':
                    if val == value:
                        accepts.append(False)
                    else:
                        accepts.append(True)
                elif cond == 'contains':
                    if value in val:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'not_contains':
                    if value in val:
                        accepts.append(False)
                    else:
                        accepts.append(True)
                elif cond == 'overlaps':
                    if any(item in val for item in value):
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'not_overlaps':
                    if any(item in val for item in value):
                        accepts.append(False)
                    else:
                        accepts.append(True)
                elif cond == 'includes':
                    if all(item in val for item in value):
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'not_includes':
                    if all(item in val for item in value):
                        accepts.append(False)
                    else:
                        accepts.append(True)
                elif cond == 'in':
                    if val in value:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'gte':
                    if val >= value:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'gt':
                    if val > value:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'lte':
                    if val <= value:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'lt':
                    if val < value:
                        accepts.append(True)
                    else:
                        accepts.append(False)
                elif cond == 'not_in':
                    if val in value:
                        accepts.append(False)
                    else:
                        accepts.append(True)

            accept = False not in accepts
            return accept


def dicom_is_view(dataset, view_shortcode, view_name):
    try:
        if "ViewPosition" in dataset:
            return dataset.ViewPosition.lower() == view_shortcode.lower()
        elif "ViewCodeSequence" in dataset:
            for viewcodesequence in dataset.ViewCodeSequence:
                if "CodeMeaning" in viewcodesequence and viewcodesequence.CodeMeaning == view_name:
                    return True
            return False
        else:
            return False
    except Exception as e:
        print(f'[dicom_is_view] Failed getting view for {dataset.SOPInstanceUID} with {e}')
        return False

def to_fixed_size(val, size=64):
    l = len(val)
    if l > size:
        return val[:size]
    elif l < size:
        pad = '0' * (size - l)
        return f'{val}{pad}'
    else:
        return val

def get_uid_from_value(val, size=64, suffix='', prefix=uid_root):
    id = hashlib.md5(val.encode('utf-8')).hexdigest()
    if not prefix.endswith('.'):
        prefix = f'{prefix}.'
    uid = f'{prefix}{int(id, 16)}{suffix}'
    return to_fixed_size(uid, size)

def generate_uid(uid, type, prefix=uid_root, di_prefix="HM-", persistent_uids=True):
    if prefix is not None and not prefix.endswith('.'):
        prefix = f'{prefix}.'

    if type in ['AccessionNumber']:
        id = hashlib.md5(uid.encode('utf-8')).hexdigest()
        high = int(id[16:], 16)
        low = int(id[:16], 16)
        mod_uid = (np.base_repr((high + low), base=36, padding=0))
        mod_uid = f'{di_prefix}{mod_uid}'
    elif type in ['PatientID']:
        id = hashlib.md5(uid.encode('utf-8')).hexdigest()
        mod_uid = (np.base_repr(int(id, 16), base=36, padding=0))
        mod_uid = f'{di_prefix}{mod_uid}'
    else:
        if persistent_uids:
            mod_uid = get_uid_from_value(uid, prefix=prefix)
        else:
            mod_uid = pydicom.uid.generate_uid(prefix=prefix)

    return mod_uid

# ----------------------------------------------------------


def dicom_get_view(ds, standardize=False):
    """
    Get view following a standardize nomenclature
    :param ds [dict]: metadata formatted as return by the tools.io.image.ImageHandler()
    """
    def _get_metadata(_v, _n):
        tmp = {k__.replace('|', '').replace(',', ''): v__ for (k__, v__) in _v.items()}.get(_n, {})
        tmp = {k__.lower(): v__ for (k__, v__) in tmp.items()}.get('value', None)
        if isinstance(tmp, list):
            tmp = tmp[0] if len(tmp) > 0 else None
        return tmp

    view = None
    if isinstance(ds, pydicom.Dataset):
        view_code_sequence = ds.get('ViewCodeSequence')
        view_code_sequence = None if not view_code_sequence or len(view_code_sequence) == 0 else view_code_sequence[0]
    else:
        view_code_sequence = _get_metadata(ds, '00540220')
    if view_code_sequence:
        if isinstance(ds, pydicom.Dataset):
            view_code_sequence = ds.get('CodeValue')
            view_code_sequence = None if not view_code_sequence or len(view_code_sequence) == 0 else view_code_sequence[
                0]
        else:
            view_code_sequence = _get_metadata(view_code_sequence, '00080100')
        if view_code_sequence:
            # https://dicom.nema.org/medical/Dicom/2018d/output/chtml/part16/sect_CID_4014.html
            mapping_vcs = {'r-10224': 'ML', # 'medio-lateral'
                           'r-10226': 'MLO', # 'medio-lateral oblique'
                           'r-10228': 'LM', # 'latero-medial'
                           'r-10230': 'LMO', # 'latero-medial oblique'
                           'r-10242': 'CC', # 'cranio-caudal'
                           'r-10244': 'FB', # 'caudo-cranial (from below)'
                           'r-102d0': 'SIO', # 'superolateral to inferomedial oblique'
                           'r-40aaa': 'ISO', # 'inferomedial to superolateral oblique'
                           'r-1024a': 'XCCL', # 'cranio-caudal exaggerated laterally'
                           'r-1024b': 'XCCM', # 'cranio-caudal exaggerated medially'
                           'g-8310':  'tissue specimen from breast'}
            view = mapping_vcs.get(view_code_sequence.lower(), None)

    if view is None:
        if isinstance(ds, pydicom.Dataset):
            view_position = ds.get('ViewPosition')
        else:
            view_position = _get_metadata(ds, '00185101')
        if view_position: view = view_position

    # Sanity formatting
    if standardize and view:
        mapping_sanity = {'medio-lateral': 'ML',
                        'medio-lateral oblique': 'MLO',
                        'latero-medial': 'LM',
                        'latero-medial oblique': 'LMO',
                        'cranio-caudal': 'CC',
                        'caudo-cranial (from below)': 'FB',
                        'superolateral to inferomedial oblique': 'SIO',
                        'inferomedial to superolateral oblique': 'ISO',
                        'cranio-caudal exaggerated laterally': 'XCCL',
                        'cranio-caudal exaggerated medially': 'XCCM',
                        'tissue specimen from breast':  'tissue specimen from breast'}
        for k, v in mapping_sanity.items():
            k = k.replace('-', '').replace('-', '').replace('(', '').replace(')', '')
            tmp = view.replace('-', '').replace('-', '').replace('(', '').replace(')', '')
            if k==tmp:
                view=v
                break
        if view not in list(mapping_sanity.values()): view=None

    return view or None


    # Get metadata
    if not view: return None
    if not standardize: return view
# ----------------------------------------------------------

def dicom_is_view_mlo(dataset):
    return dicom_is_view(dataset, 'MLO', 'medio-lateral oblique')


def dicom_is_view_cc(dataset):
    return dicom_is_view(dataset, 'CC', 'cranio-caudal')

# ----------------------------------------------------------


class DicomDataExtractor:


    @staticmethod
    def view(ds, config=None):
        out = dicom_get_view(ds, True)
        if config and out:
            if 'lowercase' in config:
                out = out.lower()

        return out


def extract_first_value_from_dicom(dicom, item, default=None):
    """
    having a dicom Dataset object and one or several items, get the values of the listed items and return the first available
    :param dicom:
    :param item:
    :return:
    """
    items = datautils.make_list(item)
    for itm in items:
        val = extract_value_from_dicom(dicom, itm, default=default)
        if val:
            return val
    return None


def extract_value_from_dicom(ds, elt, default=''):
    item = default
    if elt.startswith('extra:'):
        cfgs = elt.split(':')
        fun = getattr(DicomDataExtractor, cfgs[1])
        if len(cfgs) > 2:
            item = fun(ds, cfgs[2])
        else:
            item = fun(ds)

    elif elt.startswith('const:'):
        item = elt.replace('const:', '')

    elif ':' in elt:
        cfgs = elt.split(':')
        label = cfgs[0]
        if label in ds:

            item_raw = ds[label].value
            try:
                if isinstance(item_raw, bytes):
                    item_raw = item_raw.decode()
            except:
                pass

            if cfgs[1] == 'year':
                item = item_raw[:4]
            elif cfgs[1] == 'month':
                item = item_raw[4:4 + 2]
            elif cfgs[1] == 'day':
                item = item_raw[6:6 + 2]
            elif cfgs[1] == 'clean':
                item = item_raw
                for i in [' ', '.', ',', '=', '&', '%', '^', '/']:
                    item = f'{item}'.replace(i, '_')
            elif cfgs[1] == 'family_name':
                item = item_raw.family_name
            elif cfgs[1] == 'given_name':
                item = item.given_name
            elif cfgs[1] == 'int':
                item = str(item_raw).split('.')[0]


    else:
        if elt in ds:
            try:
                item = str(ds[elt].value)
            except:
                item = ds[elt].value
    return str(item)


def get_dicom_values_for_strategies(ds, output_strategy_elts, default='', allow_empty=True, return_map=False):
    output_strats = []
    output_strats_map = {}
    for elt in output_strategy_elts:
        if elt is not None and elt != '':
            item = default
            try:
                if '{' in elt and '}' in elt:
                    keys = elt.split('{')
                    clean_entry = elt.replace('{', '').replace('}', '')
                    vals = []
                    for k in keys:
                        if k is not None and k != '' and '}' in k:
                            key = k.split('}')[0]
                            try:
                                if ';' in key:
                                    keys = key.split(';')
                                    val = extract_first_value_from_dicom(ds, keys, default)
                                else:
                                    val = extract_value_from_dicom(ds, key, default)
                            except:
                                try:
                                    val = ds[key].value
                                except:
                                    val = default
                            vals.append(val)
                    item = '-'.join(vals)
                else:
                    item = extract_value_from_dicom(ds, elt, default)
            except Exception as e:
                pass
            if not allow_empty and item == '':
                item = default
            output_strats.append(item)
            output_strats_map[elt] = item

    if return_map:
        return output_strats_map
    else:
        return output_strats


def get_dicom_values_for_strategy(ds, strategy, default='', allow_empty=True, glue='/'):
    if callable(strategy):
        return strategy(ds)
    else:
        if isinstance(strategy, str):
            strategy = strategy.split(glue)
        return get_dicom_values_for_strategies(ds, strategy, default=default, allow_empty=allow_empty)

def get_dicom_value_for_strategy(ds, strategy, default='', allow_empty=True, glue='/'):
    val = get_dicom_values_for_strategy(ds, strategy, default=default, allow_empty=allow_empty, glue=glue)
    return glue.join(val)
