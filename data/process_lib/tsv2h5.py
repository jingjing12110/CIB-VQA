# @File  :tsv2h5.py
# @Time  :2021/4/16
# @Desc  :
import os
import sys
import csv
import base64
import h5py
import numpy as np
from tqdm import tqdm

from utils import load_json

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

source_dir = '/media/kaka/SSD1T/DataSet/vqa2/'
target_dir = '/home/kaka/Data/vqa2/'


def convert_tsv_2_h5(tsv_file, img_ids, mode='train'):
    h5_data = h5py.File(
        os.path.join(target_dir, f'imgfeat/{mode}_obj36.h5'), "w")
    
    pbar = tqdm(total=len(img_ids))
    
    with open(tsv_file) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            img_id = item['img_id']
            img_id = int(img_id.split('_')[-1])
            total_num = 0
            if img_id in img_ids:
                h5_temp = h5_data.create_group(f'{img_id}')
                
                h5_temp.create_dataset(
                    name='img_hw',
                    data=np.array([item['img_h'], item['img_w']], dtype=np.int32),
                    dtype=np.int32)

                boxes = int(item['num_boxes'])
                decode_config = [
                    ('objects_id', (boxes,), np.int64),
                    ('objects_conf', (boxes,), np.float32),
                    ('attrs_id', (boxes,), np.int64),
                    ('attrs_conf', (boxes,), np.float32),
                    ('boxes', (boxes, 4), np.float32),
                    ('features', (boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(base64.b64decode(item[key]),
                                              dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    
                    h5_temp.create_dataset(name=key, data=item[key], dtype=dtype)
                total_num += 1
                pbar.update(1)
        print(f'Number of unique image: {len(img_ids)}')
    pbar.close()


def process_vqav2():
    mode = 'train'
    file_tsv = os.path.join(
        source_dir, f'imgfeat/lxmert/{mode}2014_obj36.tsv')
    # verify tsv
    # data = load_obj_tsv(train_tsv_feat)
    
    annotations = load_json(os.path.join(
        target_dir,
        f'vqav2/raw/v2_mscoco_{mode}2014_annotations.json')
    )['annotations']
    img_ids = [a['image_id'] for a in annotations]
    img_ids = list(set(img_ids))
    print(f'Number of unique image: {len(img_ids)}')
    
    convert_tsv_2_h5(file_tsv, img_ids, mode=mode)
    
    mode = 'val'
    file_tsv = os.path.join(
        source_dir, f'imgfeat/lxmert/{mode}2014_obj36.tsv')
    
    annotations = load_json(os.path.join(
        target_dir,
        f'vqav2/raw/v2_mscoco_{mode}2014_annotations.json')
    )['annotations']
    img_ids = [a['image_id'] for a in annotations]
    img_ids = list(set(img_ids))
    print(f'Number of unique image: {len(img_ids)}')
    
    convert_tsv_2_h5(file_tsv, img_ids, mode=mode)
    
    mode = 'test2015'
    file_tsv = os.path.join(
        source_dir, f'imgfeat/lxmert/{mode}_obj36.tsv')
    
    questions = load_json(os.path.join(
        target_dir,
        f'vqav2/raw/v2_OpenEnded_mscoco_test2015_questions.json')
    )['questions']
    img_ids = [a['image_id'] for a in questions]
    img_ids = list(set(img_ids))
    print(f'Number of unique image: {len(img_ids)}')
    
    convert_tsv_2_h5(file_tsv, img_ids, mode=mode)


def convert_tsv_2_h5_ivvqa(tsv_file, id2img, h5_data, mode='train'):
    pbar = tqdm(total=len(id2img))
    
    with open(tsv_file) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            img_id = item['img_id']
            img_name = id2img[img_id].split('.')[0]
            if img_id in id2img.keys():
                h5_temp = h5_data.create_group(f'{img_name}')
                
                h5_temp.create_dataset(
                    name='img_hw',
                    data=np.array([item['img_h'], item['img_w']], dtype=np.int32),
                    dtype=np.int32)
                
                boxes = int(item['num_boxes'])
                decode_config = [
                    ('objects_id', (boxes,), np.int64),
                    ('objects_conf', (boxes,), np.float64),
                    ('attrs_id', (boxes,), np.int64),
                    ('attrs_conf', (boxes,), np.float32),
                    ('boxes', (boxes, 4), np.float32),
                    ('features', (boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    item[key] = np.frombuffer(base64.b64decode(item[key]),
                                              dtype=dtype)
                    item[key] = item[key].reshape(shape)
                    
                    h5_temp.create_dataset(name=key, data=item[key], dtype=dtype)
            pbar.update(1)
    pbar.close()



def process_ivvqa():
    # source_dir = '/home/kaka/Data/vqa2/iv_vqa/testing/'
    # file_tsv = os.path.join(
    #     source_dir, f'resnet101_faster_rcnn_iv_vqa.tsv')
    #
    # id2img = load_json(os.path.join(source_dir, 'id2img.json'))
    #
    # img_ids = id2img.keys()
    # img_ids = list(set(img_ids))
    # print(f'Number of unique image: {len(img_ids)}')
    #
    # h5_data = h5py.File(os.path.join(
    #     '/home/kaka/Data/vqa2/iv_vqa/testing/', f'iv_vqa_obj36.h5'), "w")
    # convert_tsv_2_h5_ivvqa(file_tsv, id2img, h5_data)

    # CV-VQA
    source_dir = '/home/kaka/Data/vqa2/cv_vqa/testing/'
    file_tsv = os.path.join(
        source_dir, f'resnet101_faster_rcnn_cv_vqa.tsv')

    id2img = load_json(os.path.join(source_dir, 'id2img.json'))

    img_ids = id2img.keys()
    img_ids = list(set(img_ids))
    print(f'Number of unique image: {len(img_ids)}')
    
    h5_data = h5py.File(os.path.join(
        '/home/kaka/Data/vqa2/cv_vqa/testing/', f'cv_vqa_obj36.h5'), "w")
    convert_tsv_2_h5_ivvqa(file_tsv, id2img, h5_data)


if __name__ == '__main__':
    # process_vqav2()
    process_ivvqa()

