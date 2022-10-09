# @File  :vqace_dataset.py
# @Time  :2021/5/6
# @Desc  :
import json
import os
import h5py
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer

from utils import load_json, read_txt 
from data_loader.base_dataset import _truncate_seq_pair 


class VQACEDataset(Dataset):
    def __init__(self, args, test_mode='all'):
        super().__init__()
        self.args = args
        
        self.data = load_json(f'data/vqa_ce/{test_mode}_targets.json')
        # self.data_counter = load_json(
        #     f'data/vqa_ce/counterexample_targets.json')
        # self.data_hard = load_json('data/vqa_ce/hard_targets.json')
        # self.data_easy = load_json('data/vqa_ce/easy_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            f"{datum['question_id']}": datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        self.obj_h5 = h5py.File(os.path.join(
            self.args.img_feat_root, f'val_obj36.h5'), 'r')
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['img_id']
        ques_id = f"{datum['question_id']}"
        ques = datum['question']
        
        # img feat
        img_info = self.obj_h5[f'{img_id}']
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


class VQACEVLTransDataset(VQACEDataset):
    def __init__(self, args, test_mode='all'):
        super(VQACEVLTransDataset, self).__init__(args, test_mode)
        self.que_dp = h5py.File(
            f"data/vqa_ce/que_dp_roberta.h5", 'r')
        
        self.obj_graph = h5py.File(f'data/vqav2/val_img2graph.h5', 'r')

    def __getitem__(self, item: int):
        datum = self.data[item]
    
        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']
    
        # question dependency parsing
        input_ids = self.que_dp[ques]['input_ids'][:]
        input_mask = self.que_dp[ques]['input_mask'][:]
        segment_ids = self.que_dp[ques]['segment_ids'][:]
        graph_arc = self.que_dp[ques]['graph_arc'][:]
    
        # img feat
        img_info = self.obj_h5[f'{img_id}']
        feats = img_info['features'][:].copy()
        boxes = img_info['boxes'][:].copy()
        assert len(boxes) == len(feats)
        graph_obj = self.obj_graph[img_id][:]
    
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
            return ques_id, feats, boxes, ques, target, \
                input_ids, input_mask, segment_ids, graph_arc, graph_obj
        else:
            return ques_id, feats, boxes, ques, \
                   input_ids, input_mask, segment_ids, graph_arc, graph_obj


class VQACEUNITERDataset(Dataset):
    def __init__(self, args, test_mode='all'):
        super().__init__()
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=True)  # 28996
        self.max_seq_length = 20
        
        self.data = load_json(f'data/vqa_ce/{test_mode}_targets.json')
        # self.data_counter = load_json(
        #     f'data/vqa_ce/counterexample_targets.json')
        # self.data_hard = load_json('data/vqa_ce/hard_targets.json')
        # self.data_easy = load_json('data/vqa_ce/easy_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            f"{datum['question_id']}": datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        self.obj_h5 = h5py.File(os.path.join(
            self.args.img_feat_root, f'val_obj36.h5'), 'r')
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['img_id']
        ques_id = f"{datum['question_id']}"
        ques = datum['question']

        token = self.tokenizer.tokenize(ques.strip())
        if len(token) > self.max_seq_length - 2:
            token = token[:(self.max_seq_length - 2)]
        tokens = ["[CLS]"] + token + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        attn_masks = torch.zeros(56, dtype=torch.long)
        temp = torch.ones(len(input_ids) + 36, dtype=torch.long)
        attn_masks[:temp.shape[0]] = temp

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_mask = torch.tensor(input_mask, dtype=torch.int64)
        
        # img feat
        img_info = self.obj_h5[f'{img_id}']
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
        
        pos_feat = np.zeros((36, 7), dtype=np.float32)
        pos_feat[:, :4] = boxes
        pos_feat[:, 4] = boxes[:, 2] - boxes[:, 0]
        pos_feat[:, 5] = boxes[:, 3] - boxes[:, 1]
        pos_feat[:, 6] = pos_feat[:, 4] * pos_feat[:, 5]
        
        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, feats, pos_feat, ques, target, \
                input_ids, input_mask, attn_masks
        else:
            return ques_id, feats, pos_feat, ques, \
                input_ids, input_mask, attn_masks


class VQACEViLBERTDataset(Dataset):
    def __init__(self, args, test_mode='all'):
        super().__init__()
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.max_seq_length = 20
        
        self.data = load_json(f'data/vqa_ce/{test_mode}_targets.json')
        # self.data_counter = load_json(
        #     f'data/vqa_ce/counterexample_targets.json')
        # self.data_hard = load_json('data/vqa_ce/hard_targets.json')
        # self.data_easy = load_json('data/vqa_ce/easy_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            f"{datum['question_id']}": datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        self.obj_h5 = h5py.File(os.path.join(
            self.args.img_feat_root, f'val_obj36.h5'), 'r')
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['img_id']
        ques_id = f"{datum['question_id']}"
        ques = datum['question']
        
        token = self.tokenizer.tokenize(ques.strip())
        if len(token) > self.max_seq_length - 2:
            token = token[:(self.max_seq_length - 2)]
        tokens = ["[CLS]"] + token + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(tokens)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_mask = torch.tensor(input_mask, dtype=torch.int64)
        segment_ids = torch.tensor(segment_ids, dtype=torch.int64)
        
        # img feat
        img_info = self.obj_h5[f'{img_id}']
        feats = img_info['features'][:].copy()
        boxes = img_info['boxes'][:].copy()
        assert len(boxes) == len(feats)
        
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
        boxes = boxes.copy()
        cover = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (
                    img_h * img_w)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        pos_feat = np.zeros((36, 5), dtype=np.float32)
        pos_feat[:, :4] = boxes
        pos_feat[:, 4] = cover

        image_mask = [1] * 36
        image_mask = torch.tensor(image_mask).long()
        co_attention_mask = torch.zeros((36, 20))

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, feats, pos_feat, ques, target, \
                input_ids, input_mask, segment_ids, \
                co_attention_mask, image_mask
        else:
            return ques_id, feats, pos_feat, ques, \
                   input_ids, input_mask, segment_ids, \
                   co_attention_mask, image_mask


class VQACEOSCARDataset(Dataset):
    def __init__(self, args, test_mode='all'):
        super().__init__()
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)  #
        self.max_seq_length = 60
        
        self.data = load_json(f'data/vqa_ce/{test_mode}_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            f"{datum['question_id']}": datum
            for datum in self.data
        }
        
        # Answers
        self.ans2label = json.load(open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        self.obj_h5 = h5py.File(os.path.join(
            self.args.img_feat_root, f'val_obj36.h5'), 'r')
        
        self.obj_label = read_txt(f'data/vqav2/objects_vocab.txt')
        self.id2obj = {i: self.obj_label[i] for i in range(len(self.obj_label))}
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_answers(self):
        return len(self.ans2label)
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['img_id']
        ques_id = f"{datum['question_id']}"
        ques = datum['question']
        
        # img feat
        img_info = self.obj_h5[f'{img_id}']
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
        
        pos_feat = np.zeros((36, 6), dtype=np.float32)
        pos_feat[:, :4] = boxes
        pos_feat[:, 4] = boxes[:, 2] - boxes[:, 0]
        pos_feat[:, 5] = boxes[:, 3] - boxes[:, 1]

        feats = np.concatenate((feats, pos_feat), axis=-1)  # [36, 2054]

        # object tags
        object_ids = img_info['objects_id'][:]
        object_tags = [self.id2obj[i] for i in object_ids]
        # get text info
        token = self.tokenizer.tokenize(ques.strip())

        _truncate_seq_pair(token, object_tags, self.max_seq_length - 3)

        tokens = token + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += object_tags + ["[SEP]"]
        segment_ids += [1] * (len(object_tags) + 1)
        tokens = ["[CLS]"] + tokens
        segment_ids = [0] + segment_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_mask = input_mask + [1] * 36  # add image feats mask

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_mask = torch.tensor(input_mask, dtype=torch.int64)
        segment_ids = torch.tensor(segment_ids, dtype=torch.int64)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
                # label_id = target.max(dim=0)[1].long()
            return ques_id, feats, target, input_ids, input_mask, segment_ids
        else:
            return ques_id, feats, input_ids, input_mask, segment_ids

