import argparse
from pathlib import Path
from data_process import get_cxr_paths_list, img_to_hdf5, get_cxr_path_csv, write_report_csv

'''
pathlib의 Path를 사용해 파일 경로 작업을 간편하게
data_process 모듈에서 데이터 처리 관련 함수를 불러옴
'''


def parse_args():
    parser = argparse.ArgumentParser()
    # '--csv_out_path' 인자를 받아 흉부 X선 이미지 경로를 저장할 CSV 파일 경로 지정.
    parser.add_argument('--csv_out_path', type=str, default='data/cxr_paths.csv',
                        help="Directory to save paths to all chest x-ray images in dataset.")
    # '--cxr_out_path' 인자를 받아 처리된 흉부 X선 이미지 데이터를 저장할 h5 파일 경로 설정.
    parser.add_argument('--cxr_out_path', type=str, default='data/cxr.h5',
                        help="Directory to save processed chest x-ray image data.")
    # '--dataset_type' 인자를 받아 처리할 데이터의 유형 선택. 'mimic' 또는 'chexpert-test' 선택 가능, 기본값 : 'mimic'
    parser.add_argument('--dataset_type', type=str, default='mimic', choices=['mimic', 'chexpert-test'],
                        help="Type of dataset to pre-process")
    # '--mimic_impressions_path' 인자를 받아 radiology report의 impression 섹션을 저장할 CSV 파일 경로 지정.
    parser.add_argument('--mimic_impressions_path', default='data/mimic_impressions.csv',
                        help="Directory to save extracted impressions from radiology reports.")
    # '--chest_x_ray_path' 인자를 받아 흉부 X선 이미지 데이터가 저장된 디렉토리 경로 설정.
    parser.add_argument('--chest_x_ray_path', default='./data/mimic-cxr/img_384/',
                        help="Directory where chest x-ray image data is stored. This should point to the files folder from the MIMIC chest x-ray dataset.")
    # '--radiology_reports_path' 인자를 받아 radiology report가 저장된 디렉토리 경로 설정.
    parser.add_argument('--radiology_reports_path', default='./data/mimic-cxr/files/',
                        help="Directory radiology reports are stored. This should point to the files folder from the MIMIC radiology reports dataset.")
    args = parser.parse_args()
    # TODO : mimic-cxr, cheXpert 데이터 링크
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.dataset_type == "mimic":
        # Write Chest X-ray Image HDF5 File
        # 흉부 X선 이미지 경로를 저장할 CSV 파일 작성
        get_cxr_path_csv(args.csv_out_path, args.chest_x_ray_path)
        # 흉부 X선 이미지 경로 리스트 얻음
        cxr_paths = get_cxr_paths_list(args.csv_out_path)
        # 이미지 데이터를 h5 파일로 변환, 저장
        img_to_hdf5(cxr_paths, args.cxr_out_path)
        print("Finished writing CXR HDF5 file.")
        print("Writing CSV file containing impressions for each CXR.")
        # Write CSV File Containing Impressions for each Chest X-ray
        # 각 흉부 X선에 대한 impression 섹션 CSV 파일 작성
        write_report_csv(cxr_paths, args.radiology_reports_path, args.mimic_impressions_path)
    elif args.dataset_type == "chexpert-test":
        # Get all test paths based on cxr dir
        # 흉부 X선 이미지 파일의 경로를 얻기 위해 해당 디렉토리로부터 모든 .jpg 파일 찾음
        cxr_dir = Path(args.chest_x_ray_path)
        print(cxr_dir)  # 경로 출력
        cxr_paths = list(cxr_dir.rglob("*.jpg"))  # 모든 .jpg 파일 찾음
        # 찾은 이미지 경로 출력
        print(cxr_paths)
        # 'view1'이 포함된 파일만 필터링하여 first frontal view만 남김
        cxr_paths = list(filter(lambda x: "view1" in str(x), cxr_paths))  # filter only first frontal views
        # 이미지 경로 리스트 정렬
        cxr_paths = sorted(cxr_paths)  # sort to align with groundtruth
        print(len(cxr_paths))
        # 테스트 데이터셋의 파일 개수가 500개인지 확인
        assert (len(cxr_paths) == 500)

        # 필터링된 이미지 데이터를 h5 파일로 변환 및 저장
        img_to_hdf5(cxr_paths, args.cxr_out_path)






