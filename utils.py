import os
import sys
import csv
import base64
import h5py
import pickle
import glob
import lmdb

import time
import json
import errno
from tqdm import tqdm

import numpy as np

import torch

"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:
<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
        Tokenize a sequence, converting a string s into a list of (string) tokens by
        splitting on the specified delimiter. Optionally keep or remove certain
        punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))
    
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')
    
    # if delim='' then regard the whole s as a token
    tokens = s.split(delim) if delim else [s]
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, min_token_count=1, delim=' ', punct_to_keep=None,
                punct_to_remove=None, add_special=None):
    token_to_count = {}
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
    }
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1
    
    token_to_idx = {}
    if add_special:
        for token in SPECIAL_TOKENS:
            token_to_idx[token] = len(token_to_idx)
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)
    
    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
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
                item[key].setflags(write=False)
            
            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (
        len(data), fname, elapsed_time))
    
    return data


def load_obj_h5(data_root, mode, topk=None):
    start_time = time.time()
    fname = os.path.join(data_root, f'{mode}_obj36.h5')
    finfo = os.path.join(data_root, f'{mode}_obj36_info.json')
    data_info = load_json(finfo)
    data_info_dict = {datum['img_id']: datum
                      for datum in data_info}
    print(f"Start to load Faster-RCNN detected objects from {fname}")
    data = []
    h5_file = h5py.File(fname, 'r')
    for key in h5_file.keys():
        temp = {'img_id': int(key)}
        for k in ['img_h', 'img_w', 'num_boxes']:
            temp[k] = data_info_dict[int(key)][k]
        for k in ['attrs_conf', 'attrs_id', 'boxes',
                  'features', 'objects_conf', 'objects_id']:
            temp[k] = h5_file[key].get(k)[:]
        data.append(temp)
        if topk is not None and len(data) == topk:
            break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (
        len(data), fname, elapsed_time))
    
    return data


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data, data_path, highest=False):
    protocol = 2 if highest else 0
    with open(data_path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def read_txt(file):
    with open(file, "r") as f:
        data = [line.strip('\n') for line in f.readlines()]
    return data


def write_txt(file, s):
    with open(file, 'a+') as f:
        f.write(s)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)


def convert_lmdb(lmdb_file, features_dir):
    MAP_SIZE = 1099511627776
    infiles = glob.glob(os.path.join(features_dir, "*"))
    id_list = []
    
    env = lmdb.open(lmdb_file, map_size=MAP_SIZE)
    with env.begin(write=True) as txn:
        for infile in tqdm(infiles):
            reader = np.load(infile, allow_pickle=True)
            item = {"image_id": reader.item().get("image_id")}
            img_id = str(item["image_id"]).encode()
            id_list.append(img_id)
            item["image_h"] = reader.item().get("image_height")
            item["image_w"] = reader.item().get("image_width")
            item["num_boxes"] = reader.item().get("num_boxes")
            item["boxes"] = reader.item().get("bbox")
            item["features"] = reader.item().get("features")
            txn.put(img_id, pickle.dumps(item))
        txn.put("keys".encode(), pickle.dumps(id_list))
