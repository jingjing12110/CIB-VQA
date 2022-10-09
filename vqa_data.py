# @File :vqa_data.py
# @Time :2021/5/6
# @Desc :
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

MSCOCO_IMGFEAT_ROOT = "/home/kaka/Data/vqa2/imgfeat/reserve/"
SPLIT2NAME = {
    'train': 'train',
    'val': 'minival',
    'local_val': 'nominival',
    'test': 'test'
}


class VQAv2Dataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.splits = splits.split(',')
        
        # Loading dataset
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(
                open(f"data/vqav2/{SPLIT2NAME[split]}.json")))
        print(f"Load {len(self.data):,} data from split(s) {splits}.")
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        if self.splits[0] == 'train':
            self.train_obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'train_obj36.h5'), 'r')
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'val_obj36.h5'), 'r')
        elif self.splits[0] == 'test':
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'{self.splits[0]}_obj36.h5'), 'r')
        else:
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'val_obj36.h5'), 'r')
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        
        # Get image info
        if img_id.split("_")[1] == 'train2014':
            img_info = self.train_obj_h5[f"{int(img_id.split('_')[-1])}"]
        else:
            img_info = self.obj_h5[f"{int(img_id.split('_')[-1])}"]
        
        feats = img_info['features'][:].copy()
        boxes = img_info['boxes'][:].copy()
        assert len(boxes) == len(feats)
        
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)
        
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class AdVQADataset(Dataset):
    def __init__(self, test_mode='val'):
        super().__init__()
        # Loading dataset
        self.data = []
        if test_mode == 'val':
            self.data.extend(json.load(open(
                f"data/advqa/val.json")))
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'val_obj36.h5'), 'r')
        elif test_mode == 'test':
            self.data.extend(json.load(open(
                f"data/advqa/test.json")))
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'test_obj36.h5'), 'r')
        print(f"Load {test_mode} data: {len(self.data):,}")
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, idx: int):
        """返回 idx"""
        datum = self.data[idx]
        
        ques_id = datum['question_id']
        ques = datum['sent']
        
        # Get image info
        img_id = datum['img_id']
        img_info = self.obj_h5[f"{int(img_id.split('_')[-1])}"]
        feats = img_info['features'][:].copy()
        boxes = img_info['boxes'][:].copy()
        
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)
        
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class AVQADataset(Dataset):
    def __init__(self, test_mode='val'):
        super().__init__()
        self.splits = test_mode
        self.data = []
        feat_dir = "/media/kaka/HD2T/Dataset/avqa/imgfeat"
        if test_mode == 'train':
            self.data.extend(json.load(open(
                f"data/avqa/train_all.json")))
            self.obj_h5 = h5py.File(os.path.join(
                f'{feat_dir}/train_obj100.h5'), 'r')
        elif test_mode == 'local_val':
            self.data.extend(json.load(open(
                f"data/avqa/minival.json")))
            self.obj_h5 = h5py.File(os.path.join(
                f'{feat_dir}/val_obj100.h5'), 'r')
        elif test_mode == 'val':
            self.data.extend(json.load(open(
                f"data/avqa/val.json")))
            self.obj_h5 = h5py.File(os.path.join(
                f'{feat_dir}/val_obj100.h5'), 'r')
        elif test_mode == 'test':
            self.data.extend(json.load(open(
                f"data/avqa/test.json")))
            self.obj_h5 = h5py.File(os.path.join(
                f'{feat_dir}/test_obj100.h5'), 'r')
        print(f"Load {test_mode} data: {len(self.data):,}")
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, idx: int):
        """返回 idx"""
        datum = self.data[idx]
        
        ques_id = datum['question_id']
        ques = datum['sent']
        
        # Get image info
        img_id = datum['img_id']
        img_info = self.obj_h5[f'{img_id}']
        
        feats = img_info['features'][:].copy()[:36]
        boxes = img_info['boxes'][:].copy()[:36, :4]
        
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAv2AVQADataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.splits = splits.split(',')
        
        # Loading dataset
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(
                open(f"data/vqav2/{SPLIT2NAME[split]}.json")))
            if split == "train":
                self.data.extend(json.load(open(
                    f"data/avqa/train_all.json")))
        print(f"Load {len(self.data):,} data from split(s) {splits}.")
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        if self.splits[0] == 'train':
            self.train_obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'train_obj36.h5'), 'r')
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'val_obj36.h5'), 'r')
            feat_dir = "/media/kaka/HD2T/Dataset/avqa/imgfeat"
            self.avqa_obj_h5 = h5py.File(os.path.join(
                f'{feat_dir}/train_obj100.h5'), 'r')
        elif self.splits[0] == 'test':
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'{self.splits[0]}_obj36.h5'), 'r')
        else:
            self.obj_h5 = h5py.File(os.path.join(
                MSCOCO_IMGFEAT_ROOT, f'val_obj36.h5'), 'r')
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        
        # Get image info
        if img_id.split("_")[1] == 'train2014':
            img_info = self.train_obj_h5[f"{int(img_id.split('_')[-1])}"]

            feats = img_info['features'][:].copy()
            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
            boxes = img_info['boxes'][:].copy()
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1 + 1e-5)
            np.testing.assert_array_less(-boxes, 0 + 1e-5)
        elif img_id.split("_")[1] in ['val2014', 'test2015']:
            img_info = self.obj_h5[f"{int(img_id.split('_')[-1])}"]

            feats = img_info['features'][:].copy()
            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
            boxes = img_info['boxes'][:].copy()
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1 + 1e-5)
            np.testing.assert_array_less(-boxes, 0 + 1e-5)
        else:
            img_info = self.avqa_obj_h5[f'{img_id}']
            feats = img_info['features'][:].copy()[:36]
            boxes = img_info['boxes'][:].copy()[:36, :4]
        
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

