# @File  :vqar_dataset.py
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


class VQARephraseDataset(Dataset):
    """simple data format of LXMERT
    """
    
    def __init__(self, args, test_mode='vqa-rephrasings'):
        super().__init__()
        self.args = args
        
        data = load_json(
            f'data/vqa_rep/val2014_humans_{test_mode}_targets.json')
        # self.data = self.data[::5]

        # answers = [a for d in data for a in d['label']]
        answers = [list(d['label'].keys())[0] for d in data]
        from collections import Counter
        answer = Counter(answers)
        # top100 = [d[0] for d in answer.most_common(30)]
        low_top100 = [d[0] for d in answer.most_common(len(answer))[:]]
        # total_ans_num = answer.most_common(50)[-1][1]
        total_ans_num = 100

        # answer_show_top100 = [
        #     # high frequency
        #     'yes', 'no',
        #     '1', '2', '3',
        #     'white',
        #     '4', '0',
        #     'red', 'blue', 'black',
        #     '5',
        #     'brown', 'green', 'yellow',
        #     '6',
        #     'gray', 'orange',
        #     '7', '8',
        #     'unknown', 'nothing', 'left', 'right',
        #     '10',
        #     'tan', 'pink', 'wood', 'many',
        #     '9',
        #     'tennis', 'not sure',
        #     '12'
        #     'pizza', 'black and white', 'outside', 'man',
        #     '20',
        #     'water', 'food', 'baseball', 'grass',
        #     "don't know", 'silver', 'frisbee', 'table', 'bathroom',
        #     'kitchen', 'lot', 'beige', 'cat',
        # ]
        #
        # # low frequency
        # answer_show_100 = ['teacher',
        #                    'slow down',
        #                    'hammock',
        #                    'on car',
        #                    'on suitcase',
        #                    '12:10',
        #                    'ducati',
        #                    'gatorade',
        #                    'db',
        #                    'zipper',
        #                    'bin',
        #                    'hawaiian',
        #                    'water skis',
        #                    'horse racing',
        #                    'sunrise',
        #                    'radiator',
        #                    'catch ball',
        #                    'riding horses',
        #                    'cameras',
        #                    'tongs',
        #                    "baby's breath",
        #                    'bus driver',
        #                    'bird feeder',
        #                    'lemonade',
        #                    'recessed',
        #                    '2008',
        #                    'feta',
        #                    'toothpaste',
        #                    '6:20',
        #                    'in box',
        #                    'napkins',
        #                    'victoria',
        #                    'wall st',
        #                    'watch tv',
        #                    'makeup',
        #                    '61',
        #                    'burton',
        #                    'dog show',
        #                    '6:40',
        #                    'trucks',
        #                    'first base',
        #                    '1:25',
        #                    'half full',
        #                    'apple and banana',
        #                    'sailboats',
        #                    'harry potter',
        #                    'size',
        #                    'horizontal',
        #                    'meow',
        #                    'marshmallows',
        #                    'j',
        #                    'mane',
        #                    'alligator',
        #                    'ladybug',
        #                    'turtle',
        #                    "st patrick's day",
        #                    'almonds',
        #                    'space',
        #                    'john',
        #                    'telling time',
        #                    'to catch frisbee',
        #                    'mountain dew',
        #                    'listening to music',
        #                    'hammer time',
        #                    'clams',
        #                    'snowsuit',
        #                    'strike',
        #                    'clydesdale',
        #                    'at&t',
        #                    '2:05',
        #                    'grassy',
        #                    'broom',
        #                    'stability',
        #                    'lighting',
        #                    'petting horse',
        #                    'license plate',
        #                    'knee pads',
        #                    'jumped',
        #                    'visilab',
        #                    'brewers',
        #                    'cluttered',
        #                    'honey',
        #                    'riding elephant',
        #                    'compaq',
        #                    'diet coke',
        #                    'wrist',
        #                    'thomas',
        #                    'shower head',
        #                    'best buy',
        #                    'national express',
        #                    'wildebeest',
        #                    'new orleans',
        #                    'on tree',
        #                    '4 way',
        #                    'blending',
        #                    'potato salad',
        #                    '101',
        #                    'tiara',
        #                    'sas',
        #                    'worms']
        #
        # answer_show_200 = ['klm',
        #                    'toasted',
        #                    'gazebo',
        #                    'bending',
        #                    'toilet brush',
        #                    'amazon',
        #                    "women's",
        #                    'charging',
        #                    'air france',
        #                    'wide',
        #                    'futon',
        #                    '2:15',
        #                    'emergency',
        #                    'facebook',
        #                    'sauerkraut',
        #                    'easy',
        #                    'descending',
        #                    'tow',
        #                    'wheelchair',
        #                    'to catch ball',
        #                    'onion rings',
        #                    'toiletries',
        #                    'thoroughbred',
        #                    '8:00',
        #                    'parmesan cheese',
        #                    'banana peel',
        #                    'barber shop',
        #                    'hiking',
        #                    'mississippi',
        #                    'over easy',
        #                    'coaster',
        #                    'motel',
        #                    'joshua',
        #                    'salt and pepper',
        #                    'chest',
        #                    'no man',
        #                    'in vase',
        #                    'greyhound',
        #                    'guitar hero',
        #                    'paw',
        #                    'numbers',
        #                    'sony ericsson',
        #                    'no water',
        #                    'lanes',
        #                    'drying',
        #                    'nasa',
        #                    '9:50',
        #                    '9:55',
        #                    'mickey mouse',
        #                    'daffodils',
        #                    'savory',
        #                    'dc',
        #                    'oriental',
        #                    'dinosaur',
        #                    '4:05',
        #                    'walgreens',
        #                    'ladder',
        #                    'christmas tree',
        #                    '12:28',
        #                    'jal',
        #                    'korean air',
        #                    'us airways',
        #                    'conference room',
        #                    'spray paint',
        #                    '7:10',
        #                    'flashlight',
        #                    'lifeguard',
        #                    'dreadlocks',
        #                    'golden gate',
        #                    'pizza box',
        #                    "they aren't",
        #                    'practice',
        #                    '68',
        #                    'around neck',
        #                    '7:25',
        #                    'lacoste',
        #                    'virgin atlantic',
        #                    'pacifier',
        #                    'hanger',
        #                    'no train',
        #                    'suspenders',
        #                    'throw frisbee',
        #                    'index',
        #                    '5:18',
        #                    'mariners',
        #                    'tigers',
        #                    'tv stand',
        #                    'style',
        #                    'fork and spoon',
        #                    '10:40',
        #                    'nose',
        #                    'chevron',
        #                    'coins',
        #                    'cherries',
        #                    'us open',
        #                    'bald',
        #                    'top hat',
        #                    'ping pong',
        #                    'coffee maker',
        #                    'on horse',
        #                    'teacher',
        #                    'slow down',
        #                    'hammock',
        #                    'on car',
        #                    'on suitcase',
        #                    '12:10',
        #                    'ducati',
        #                    'gatorade',
        #                    'db',
        #                    'zipper',
        #                    'bin',
        #                    'hawaiian',
        #                    'water skis',
        #                    'horse racing',
        #                    'sunrise',
        #                    'radiator',
        #                    'catch ball',
        #                    'riding horses',
        #                    'cameras',
        #                    'tongs',
        #                    "baby's breath",
        #                    'bus driver',
        #                    'bird feeder',
        #                    'lemonade',
        #                    'recessed',
        #                    '2008',
        #                    'feta',
        #                    'toothpaste',
        #                    '6:20',
        #                    'in box',
        #                    'napkins',
        #                    'victoria',
        #                    'wall st',
        #                    'watch tv',
        #                    'makeup',
        #                    '61',
        #                    'burton',
        #                    'dog show',
        #                    '6:40',
        #                    'trucks',
        #                    'first base',
        #                    '1:25',
        #                    'half full',
        #                    'apple and banana',
        #                    'sailboats',
        #                    'harry potter',
        #                    'size',
        #                    'horizontal',
        #                    'meow',
        #                    'marshmallows',
        #                    'j',
        #                    'mane',
        #                    'alligator',
        #                    'ladybug',
        #                    'turtle',
        #                    "st patrick's day",
        #                    'almonds',
        #                    'space',
        #                    'john',
        #                    'telling time',
        #                    'to catch frisbee',
        #                    'mountain dew',
        #                    'listening to music',
        #                    'hammer time',
        #                    'clams',
        #                    'snowsuit',
        #                    'strike',
        #                    'clydesdale',
        #                    'at&t',
        #                    '2:05',
        #                    'grassy',
        #                    'broom',
        #                    'stability',
        #                    'lighting',
        #                    'petting horse',
        #                    'license plate',
        #                    'knee pads',
        #                    'jumped',
        #                    'visilab',
        #                    'brewers',
        #                    'cluttered',
        #                    'honey',
        #                    'riding elephant',
        #                    'compaq',
        #                    'diet coke',
        #                    'wrist',
        #                    'thomas',
        #                    'shower head',
        #                    'best buy',
        #                    'national express',
        #                    'wildebeest',
        #                    'new orleans',
        #                    'on tree',
        #                    '4 way',
        #                    'blending',
        #                    'potato salad',
        #                    '101',
        #                    'tiara',
        #                    'sas',
        #                    'worms']
        self.data = []
        count_key = {}
        for datum in data:
            label_key = list(datum['label'].items())[0][0]
            if label_key in low_top100:
                if label_key in count_key:
                    if count_key[label_key] <= total_ans_num:
                        datum['label'] = {
                            label_key: list(datum['label'].items())[0][1]
                        }
                        self.data.append(datum)

                    count_key[label_key] += 1
                else:
                    count_key[label_key] = 1
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
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
        
        img_id = datum['image_id']
        ques_id = datum['question_id']
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
            # label = [a for a in label.items()][0]
            # ans, score = label[0], label[1]
            # target[self.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQARephraseVLTransDataset(VQARephraseDataset):
    """data format of our model (Graph Transformer)
    """
    
    def __init__(self, args, test_mode='vqa-rephrasings'):
        super(VQARephraseVLTransDataset, self).__init__(args, test_mode=test_mode)
        
        self.que_dp = h5py.File(
            f"data/vqa_rep/vqa_rep_que_dp_roberta.h5", 'r')
        
        # VQA-Rep images origin from VQA v2.0 validation set
        self.obj_graph = h5py.File(f'data/vqav2/val_img2graph.h5', 'r')
    
    def __getitem__(self, item: int):
        datum = self.data[item]
        
        img_id = datum['image_id']
        ques_id = datum['question_id']
        ques = datum['question']
        
        # question dependency parsing
        input_ids = self.que_dp[ques_id]['input_ids'][:]
        input_mask = self.que_dp[ques_id]['input_mask'][:]
        segment_ids = self.que_dp[ques_id]['segment_ids'][:]
        graph_arc = self.que_dp[ques_id]['graph_arc'][:]
        
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


class VQARephraseUNITERDataset(Dataset):
    """data format of UNITER series
    """
    
    def __init__(self, args, test_mode='vqa-rephrasings'):
        super().__init__()
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=True)  # 28996
        self.max_seq_length = 20
        
        self.data = load_json(
            f'data/vqa_rep/val2014_humans_{test_mode}_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
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
        
        img_id = datum['image_id']
        ques_id = datum['question_id']
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


class VQARephraseViLBERTDataset(Dataset):
    """data format of VilBERT
    """
    
    def __init__(self, args, test_mode='vqa-rephrasings'):
        super().__init__()
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)  # 30522
        self.max_seq_length = 20
        
        self.data = load_json(
            f'data/vqa_rep/val2014_humans_{test_mode}_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
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
        
        img_id = datum['image_id']
        ques_id = datum['question_id']
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


class VQARephraseOSCARDataset(Dataset):
    """simple data format of OSCAR 
    """
    
    def __init__(self, args, test_mode='vqa-rephrasings'):
        super().__init__()
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)  #
        self.max_seq_length = 60
        
        self.data = load_json(
            f'data/vqa_rep/val2014_humans_{test_mode}_targets.json')
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
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
        
        img_id = datum['image_id']
        ques_id = datum['question_id']
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


if __name__ == '__main__':
    from baseline.param import args
    from torch.utils.data.dataloader import DataLoader
    from tqdm import tqdm
    
    dataset = VQARephraseDataset(args, test_mode='original')
    data_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )
    for i, data_tuple in tqdm(
            enumerate(data_loader), total=len(data_loader), ncols=80):
        que_id = data_tuple[0]
