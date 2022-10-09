# @File  :base_dataset.py
# @Time  :2021/5/8
# @Desc  :
import json
import collections
import numpy as np
import torch
from prefetch_generator import BackgroundGenerator


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


# ************************************************************************
# tools for data loading
class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        
        self.preload()
    
    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
    
    def next(self):
        batch = self.batch
        self.preload()
        return batch


class _RepeatSampler(object):
    """ 一直repeat的sampler """
    
    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ 多epoch训练时，DataLoader对象不用重新建立线程和batch_sampler对象，
    以节约每个epoch的初始化时间 """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
    
    def __len__(self):
        return len(self.batch_sampler.sampler)
    
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


# ************************************************************************
# VQA Evaluator
class VQAEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def evaluate(self, qid2ans: dict):
        score = 0.
        for que_id, ans in qid2ans.items():
            datum = self.dataset.id2datum[que_id]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(qid2ans)
    
    @staticmethod
    def dump_result(qid2ans: dict, path):
        """Dump results to a json file, which could be submitted to
        the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param qid2ans: dict of qid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in qid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


def compute_bbox_overlapping(boxes, num_box=36):
    relation_mask = np.zeros((num_box, num_box))
    for i in range(num_box):
        for j in range(i + 1, num_box):
            # if there is no overlap between two bounding box
            if boxes[i, 0] > boxes[j, 2] or boxes[j, 0] > boxes[i, 2] or \
                    boxes[i, 1] > boxes[j, 3] or boxes[j, 1] > boxes[i, 3]:
                pass
            else:
                relation_mask[i, j] = relation_mask[j, i] = 1
    relation_mask = torch.from_numpy(relation_mask).long()
    return relation_mask


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
