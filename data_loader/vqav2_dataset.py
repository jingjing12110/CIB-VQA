# @File :vqav2_dataset.py
# @Time :2021/5/6
# @Desc :
import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
# from sklearn.neighbors import kneighbors_graph

from baseline.param import args
from utils import load_json, read_txt


# The path to data and image features.
MSCOCO_IMGFEAT_ROOT = args.img_feat_root


SPLIT2NAME = {
    'train': 'train',
    'val': 'nominival',
    'minival': 'minival',  # local testing
    'test': 'test'
}


"""
A VQA data example in json file:
    {
        "answer_type": "other",
        "img_id": "COCO_train2014_000000458752",
        "label": {
            "net": 1
        },
        "question_id": 458752000,
        "question_type": "what is this",
        "sent": "What is this photo taken looking through?"
    }
"""


class VQAv2Dataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.splits = splits.split(',')
        
        # Loading dataset
        self.data_org = []
        for split in self.splits:
            self.data_org.extend(json.load(
                open(f"data/vqav2/{SPLIT2NAME[split]}.json")))
        print("Load %d data from split(s) %s." % (
            len(self.data_org), splits))
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data_org
        }
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        if self.splits[0] == 'minival':
            self.splits[0] = 'val'
        self.obj_h5 = h5py.File(os.path.join(
            MSCOCO_IMGFEAT_ROOT, f'{self.splits[0]}_obj36.h5'), 'r')
        
        # Only kept the data with loaded image features
        self.data = []
        for datum in self.data_org:
            img_id = str(int(datum['img_id'].split('_')[-1]))
            if img_id in self.obj_h5.keys():
                datum['img_id'] = img_id
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
    
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
        img_info = self.obj_h5[f'{img_id}']
        # obj_num = img_info['num_boxes']
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


class VQAv2LTransDataset(Dataset):
    def __init__(self, splits, k_neighbors=8):
        super().__init__()
        self.splits = splits.split(',')
        self.k_neighbors = k_neighbors
        
        # Loading dataset
        self.data_org = []
        for split in self.splits:
            self.data_org.extend(json.load(
                open(f"data/vqav2/{SPLIT2NAME[split]}.json")))

            self.que_dp = h5py.File(
                f"data/vqav2/{SPLIT2NAME[split]}_que_dp_roberta.h5", 'r')
        
        print("Load %d data from split(s) %s." % (
            len(self.data_org), splits))
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data_org
        }
        
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features
        if self.splits[0] == 'minival':
            self.splits[0] = 'val'
        self.obj_h5 = h5py.File(os.path.join(
            MSCOCO_IMGFEAT_ROOT, f'{self.splits[0]}_obj36.h5'), 'r')
        
        # Loading pre-defined objects relations
        self.obj_graph = h5py.File(
            f'data/vqav2/{self.splits[0]}_img2graph.h5', 'r')
        
        # Only kept the data with loaded image features
        self.data = []
        for datum in self.data_org:
            img_id = str(int(datum['img_id'].split('_')[-1]))
            if img_id in self.obj_h5.keys():
                datum['img_id'] = img_id
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
    
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
        
        # question dependency parsing 
        input_ids = self.que_dp[ques]['input_ids'][:]
        input_mask = self.que_dp[ques]['input_mask'][:]
        segment_ids = self.que_dp[ques]['segment_ids'][:]
        graph_arc = self.que_dp[ques]['graph_arc'][:]
        
        # Get image info
        img_info = self.obj_h5[f'{img_id}']
        # obj_num = img_info['num_boxes']
        feats = img_info['features'][:].copy()
        boxes = img_info['boxes'][:].copy()
        assert len(boxes) == len(feats)
        
        # Random Graph: 随机初始化
        # graph_obj = torch.randint(0, 4, (36, 36)).long()
        # graph_obj = graph_obj.tril(-1) + graph_obj.triu(1)
        
        # pre-defined graph: 四种类别
        graph_obj = self.obj_graph[img_id][:]

        # KNN-Graph: features特征图
        # graph_obj = kneighbors_graph(feats, self.k_neighbors, metric='cosine')
        # graph_obj = np.array(graph_obj.todense(), dtype=np.int64)
        
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


class VQAv2VLGTransDataset(Dataset):
    def __init__(self, splits):
        super().__init__()
        self.splits = splits.split(',')
        
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)  #
        self.max_seq_length = 60
        
        # Loading dataset
        self.data_org = []
        for split in self.splits:
            self.data_org.extend(json.load(
                open(f"data/vqav2/{SPLIT2NAME[split]}.json")))
            self.que_dp = h5py.File(
                f"data/vqav2/{SPLIT2NAME[split]}_que_dp_roberta.h5", 'r')
        print("Load %d data from split(s) %s." % (
            len(self.data_org), splits))
        
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data_org
        }
        # Answers
        self.ans2label = json.load(
            open("data/vqav2/trainval_ans2label.json"))
        self.label2ans = json.load(open(
            "data/vqav2/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        
        # Loading detection features to img_data
        if self.splits[0] == 'minival':
            self.splits[0] = 'val'
        self.obj_h5 = h5py.File(os.path.join(
            MSCOCO_IMGFEAT_ROOT, f'{self.splits[0]}_obj36.h5'), 'r')
        
        # object tags
        self.obj_label = read_txt(f'data/vqav2/objects_vocab.txt')
        self.id2obj = {i: self.obj_label[i] for i in range(len(self.obj_label))}
        
        self.obj_graph = h5py.File(
            f'data/vqav2/{self.splits[0]}_img2graph.h5', 'r')
        
        # Only kept the data with loaded image features
        self.data = []
        for datum in self.data_org:
            img_id = str(int(datum['img_id'].split('_')[-1]))
            if img_id in self.obj_h5.keys():
                datum['img_id'] = img_id
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
    
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
        
        # question dependency parsing
        input_ids = self.que_dp[ques]['input_ids'][:]
        input_mask = self.que_dp[ques]['input_mask'][:]
        segment_ids = self.que_dp[ques]['segment_ids'][:]
        graph_arc = self.que_dp[ques]['graph_arc'][:]
        
        # Get image info
        img_info = self.obj_h5[f'{img_id}']
        feats = img_info['features'][:].copy()  # fixed length
        boxes = img_info['boxes'][:].copy()
        assert len(boxes) == len(feats)
        graph_obj = self.obj_graph[img_id][:]
        # graph_obj = torch.randint(0, 4, (36, 36)).long()
        # graph_obj = graph_obj.tril(-1) + graph_obj.triu(1)
        
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
        boxes = boxes.copy()
        cover = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (
                    img_h * img_w)
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
                input_ids, input_mask, segment_ids, \
                graph_arc, graph_obj
        else:
            return ques_id, feats, boxes, ques, \
                   input_ids, input_mask, segment_ids, \
                   graph_arc, graph_obj


if __name__ == '__main__':
    tset = VQAv2Dataset(splits='val')
    # evaluator = VQAEvaluator(tset)
