import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from skimage import io, exposure


class CheXpertDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.patient_id_to_meta_info = {}
        pattern = r"patient(\d+)"  # extract patient id
        for idx, d in data.iterrows():
            patient_id = re.search(pattern, d.Path).group(1)
            path = f"data/CheXpert/{d.Path.replace('CheXpert-v1.0/valid/', 'val/')}"
            if patient_id in self.patient_id_to_meta_info:  # another view of an existing patient
                assert path not in self.patient_id_to_meta_info[patient_id]['image_paths']
                self.patient_id_to_meta_info[patient_id]['image_paths'].append(path)
            else:  # First time patient
                diseases = {}
                # Hardcoded for robustness and consistency
                disease_keys = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
                for key, value in d.items():
                    if key in disease_keys:
                        diseases[key] = value
                disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
                self.patient_id_to_meta_info[patient_id] = {'image_paths': [path], 'disease_keys': disease_keys, 'disease_values': disease_values}
        self.patient_id_to_meta_info = sorted(self.patient_id_to_meta_info.items())

    def __len__(self):
        return len(self.patient_id_to_meta_info)

    def __getitem__(self, idx):
        patient_id, meta_info = self.patient_id_to_meta_info[idx]
        image_paths = meta_info['image_paths']
        keys = meta_info['disease_keys']
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        return image_paths, labels, keys

class CheXpertTestDataset(Dataset):
    def __init__(self, csv_file, transforms= None):
        data = pd.read_csv(csv_file)
        self.patient_id_to_meta_info = {}
        pattern = r"patient(\d+)"  # extract patient id
        for idx, d in data.iterrows():
            patient_id = re.search(pattern, d.Path).group(1)
            # print(d.Path)
            path = f"data/CheXpert/{d.Path.replace('CheXpert-v1.0/valid/', 'val/')}"

            if patient_id in self.patient_id_to_meta_info:  # another view of an existing patient
                assert path not in self.patient_id_to_meta_info[patient_id]['image_paths']
                self.patient_id_to_meta_info[patient_id]['image_paths'].append(path)
            else:  # First time patient
                diseases = {}
                # Hardcoded for robustness and consistency
                disease_keys = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
                for key, value in d.items():
                    if key in disease_keys:
                        diseases[key] = value
                disease_values = [diseases[key] for key in disease_keys]  # same order as disease_keys
                self.patient_id_to_meta_info[patient_id] = {'image_paths': [path], 'disease_keys': disease_keys, 'disease_values': disease_values}
        self.patient_id_to_meta_info = sorted(self.patient_id_to_meta_info.items())
        self.transform = transforms

    def __len__(self):
        return len(self.patient_id_to_meta_info)

    def __getitem__(self, idx):
        patient_id, meta_info = self.patient_id_to_meta_info[idx]
        image_paths = meta_info['image_paths']
        image = io.imread(image_paths[0])
        image = remap_to_uint8(image)
        # image = exposure.equalize_adapthist(np.array(image), clip_limit=0.03)
        image = Image.fromarray(image).convert("L")
        # print(f" image size = {image.size}")
        labels = torch.tensor(meta_info['disease_values'], dtype=torch.float)
        keys = meta_info['disease_keys']


        # image.save(f'../image_sample/image_.jpg')
        if self.transform:
            image = self.transform(image)

        # print(image_paths)
        return image, labels, keys


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)

if __name__ == "__main__":
    dataset = CheXpertTestDataset('data/CheXpert/test_labels.csv')
    print(len(dataset))
    for i ,(image, labels) in enumerate(dataset):
        print(i, image, labels)
        if i == 10:
            break