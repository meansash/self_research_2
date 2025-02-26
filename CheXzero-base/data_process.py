import os

import glob
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import h5py
import cv2
from typing import *
from pathlib import Path

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

'''
Library : 데이터 처리, 이미지 처리, h5 파일 처리, 경로 작업, 
데이터 타입 명시 및 진행 표시를 위한 모듈 등
'''


# CSV 파일을 읽어서 Pandas DataFrame으로 반환
def load_data(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe


# CSV 파일에서 'Path' 열을 가져와 흉부 X선 경로 리스트로 반환
def get_cxr_paths_list(filepath):
    dataframe = load_data(filepath)
    cxr_paths = dataframe['Path']
    return cxr_paths


'''
This function resizes and zero pads image 
'''


def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)
    # 기존 이미지 크기에서 가장 긴 쪽을 기준으로 비율을 계산한 뒤 새로운 크기로 이미지 크기 조정
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    # 크기가 조정된 이미지를 새 이미지의 중앙에 배치하여 원하는 크기(320x320)에 맞게 0으로 패딩 처리
    return new_img


def img_to_hdf5(cxr_paths: List[Union[str, Path]], out_filepath: str, resolution=320):
    """
    Convert directory of images into a .h5 file given paths to all
    images.
    """
    dset_size = len(cxr_paths)
    failed_images = []
    with h5py.File(out_filepath, 'w') as h5f:
        # h5 파일을 열고 'cxr'이라는 이름의 데이터셋 생성. 크기는 (이미지 수, 해상도, 해상도)
        img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution))
        for idx, path in enumerate(tqdm(cxr_paths)):
            try:
                # read image using cv2
                img = cv2.imread(str(path))
                # convert to PIL Image object
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                # preprocess
                img = preprocess(img_pil, desired_size=resolution)
                img_dset[idx] = img
            except Exception as e:
                failed_images.append((path, e))
    print(f"{len(failed_images)} / {len(cxr_paths)} images failed to be added to h5.", failed_images)


# 주어진 디렉토리에서 .jpg 파일들을 찾아 경로 리스트로 반환
def get_files(directory):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        for file in filenames:
            if file.endswith(".jpg"):
                files.append(os.path.join(dirpath, file))
    return files


# 디렉토리 내 모든 .jpg 파일 경로를 CSV 파일로 저장하는 함수
def get_cxr_path_csv(out_filepath, directory):
    files = get_files(directory)
    file_dict = {"Path": files}
    df = pd.DataFrame(file_dict)
    df.to_csv(out_filepath, index=False)


# 특정 섹션이 시작되는 줄 번호를 찾는 함수 (여기서는 impression section)
def section_start(lines, section=' IMPRESSION'):
    for idx, line in enumerate(lines):
        if line.startswith(section):
            return idx
    return -1


# 섹션이 끝나는 지점을 찾는 함수
def section_end(lines, section_start):
    num_lines = len(lines)


def getIndexOfLast(l, element):
    """ Get index of last occurence of element
    @param l (list): list of elements
    @param element (string): element to search for
    @returns (int): index of last occurrence of element
    """
    i = max(loc for loc, val in enumerate(l) if val == element)
    return i


def write_report_csv(cxr_paths, txt_folder, out_path):
    imps = {"filename": [], "impression": []}
    txt_reports = []
    for cxr_path in tqdm(cxr_paths):
        tokens = cxr_path.split('/')

        study_num = tokens[-2]
        patient_num = tokens[-3]
        patient_group = patient_num[0:3]
        # 각 흉부 X선 경로에 대응하는 radiology report 경로 생성
        txt_report = txt_folder + patient_group + '/' + patient_num + '/' + study_num + '.txt'

        filename = study_num + '.txt'
        f = open(txt_report, 'r')
        s = f.read()
        s_split = s.split()
        # 'impression' 섹션을 찾아서 그 이후의 텍스트 추출
        if "IMPRESSION:" in s_split:
            begin = getIndexOfLast(s_split, "IMPRESSION:") + 1
            end = None
            end_cand1 = None
            end_cand2 = None
            # remove recommendation(s) and notification
            if "RECOMMENDATION(S):" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION(S):")
            elif "RECOMMENDATION:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATION:")
            elif "RECOMMENDATIONS:" in s_split:
                end_cand1 = s_split.index("RECOMMENDATIONS:")

            if "NOTIFICATION:" in s_split:
                end_cand2 = s_split.index("NOTIFICATION:")
            elif "NOTIFICATIONS:" in s_split:
                end_cand2 = s_split.index("NOTIFICATIONS:")

            if end_cand1 and end_cand2:
                end = min(end_cand1, end_cand2)
            elif end_cand1:
                end = end_cand1
            elif end_cand2:
                end = end_cand2

            if end == None:
                imp = " ".join(s_split[begin:])
            else:
                imp = " ".join(s_split[begin:end])
        else:
            imp = 'NO IMPRESSION'

        imps["impression"].append(imp)
        imps["filename"].append(filename)

    df = pd.DataFrame(data=imps)
    df.to_csv(out_path, index=False)

