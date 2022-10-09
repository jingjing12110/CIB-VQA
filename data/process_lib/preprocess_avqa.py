# @File :preprocess_avqa.py
# @Time :2021/8/20
# @Desc :
import os
from tqdm import tqdm
import numpy as np
import h5py
from tqdm import tqdm

from utils import load_json, save_json, read_txt
from data.process_lib.annotation_process import preprocess_answer, get_score

source_dir = '/media/kaka/HD2T/Dataset/avqa/anns/combine'
target_dir = 'data/avqa/anns/'


def compute_test_target(answers_dset, ans2label, dataset='avqa'):
    target = []
    tbar = tqdm(total=len(answers_dset), ncols=80)
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = preprocess_answer(answer['answer'])
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        
        label = {}
        for answer in answer_count:
            if answer not in ans2label:
                continue
            # score = get_score(answer_count[answer])
            label[answer] = get_score(answer_count[answer])
        
        # if label:
        if dataset == 'avqa':
            target.append({
                'question_id': ans_entry['question_id'],
                'question': ans_entry['question'],
                'image_id': ans_entry['image_id'],
                'image_url': ans_entry['image_url'],
                'label': label,
                'answer_type': ans_entry['answer_type']
            })
        
        tbar.update(1)
    tbar.close()
    print(f'len target: {len(target)}')
    return target


def combine_ann_que():
    mode = 'r1_train'
    verify = 'verified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    anns_r1_train = load_json(ann_file)
    print(f'len of anns: {len(anns_r1_train)}')
    
    verify = 'unverified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    anns_r1_train_unverified = load_json(ann_file)
    print(f'len of anns: {len(anns_r1_train_unverified)}')
    
    # r2
    mode = 'r2_train'
    verify = 'verified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    anns_r2_train = load_json(ann_file)
    print(f'len of anns: {len(anns_r2_train)}')
    
    verify = 'unverified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    anns_r2_train_unverified = load_json(ann_file)
    print(f'len of anns: {len(anns_r2_train_unverified)}')
    
    # r3
    mode = 'r3_train'
    verify = 'verified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    anns_r3_train = load_json(ann_file)
    print(f'len of anns: {len(anns_r3_train)}')
    
    verify = 'unverified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    anns_r3_train_unverified = load_json(ann_file)
    print(f'len of anns: {len(anns_r3_train_unverified)}')
    
    train_all = anns_r1_train + anns_r1_train_unverified + anns_r2_train + \
                anns_r2_train_unverified + anns_r3_train + \
                anns_r3_train_unverified
    save_json(train_all, os.path.join(source_dir, f"train_all_raw_anns.json"))
    
    mode = 'r1_val'
    ann_file = os.path.join(source_dir, f"v1_avqa_{mode}_anns.json")
    anns_r1_val = load_json(ann_file)
    print(f'len of anns: {len(anns_r1_val)}')
    
    mode = 'r2_val'
    ann_file = os.path.join(source_dir, f"v1_avqa_{mode}_anns.json")
    anns_r2_val = load_json(ann_file)
    print(f'len of anns: {len(anns_r2_val)}')
    
    mode = 'r3_val'
    ann_file = os.path.join(source_dir, f"v1_avqa_{mode}_anns.json")
    anns_r3_val = load_json(ann_file)
    print(f'len of anns: {len(anns_r3_val)}')
    
    val_all = anns_r1_val + anns_r2_val + anns_r3_val
    save_json(val_all, os.path.join(source_dir, f"val_all_raw_anns.json"))


def format_annotation():
    mode = 'train'
    anns = load_json(f"{source_dir}/train_all_raw_anns.json")
    
    ans2label = load_json('data/vqav2/trainval_ans2label.json')
    # compute target
    target = compute_test_target(
        anns,
        ans2label=ans2label,
        dataset='avqa'
    )
    print(f'len of target: {len(target)}')
    # save_json(target, os.path.join(
    #     target_dir, f'v1_avqa_{mode}_{verify}_targets.json'))
    save_json(target, os.path.join(
        target_dir, f'v1_avqa_{mode}_targets.json'))


def main():
    mode = 'r1_train'
    verify = 'verified'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_annotations.json")
    anns = load_json(ann_file)['annotations']
    que_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_{verify}_questions.json")
    ques = load_json(que_file)['questions']
    print(f"len anns: {len(anns)}\n "
          f"len ques: {len(ques)}")
    
    qid2ques = {que['question_id']: que for que in ques}
    # combine
    for ann in anns:
        que = qid2ques[ann['question_id']]
        ann['question'] = que['question']
        ann['image_url'] = que['image_url']
    save_json(
        anns,
        os.path.join(target_dir, f"v1_avqa_{mode}_{verify}_anns.json")
    )
    
    # for val
    mode = 'r3_val'
    ann_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_annotations.json")
    anns = load_json(ann_file)['annotations']
    que_file = os.path.join(
        source_dir, f"v1_avqa_{mode}_questions.json")
    ques = load_json(que_file)['questions']
    print(f"len anns: {len(anns)}\n "
          f"len ques: {len(ques)}")
    qid2ques = {que['question_id']: que for que in ques}
    # combine
    for ann in anns:
        que = qid2ques[ann['question_id']]
        ann['question'] = que['question']
        ann['image_url'] = que['image_url']
    save_json(
        anns,
        os.path.join(target_dir, f"v1_avqa_{mode}_anns.json")
    )
    
    # mode = 'r1_test'
    # # ann_file = os.path.join(
    # #     source_dir, f"v1_avqa_{mode}_annotations.json")
    # # anns = load_json(ann_file)['annotations']
    # que_file = os.path.join(
    #     source_dir, f"v1_avqa_{mode}_questions.json")
    # ques = load_json(que_file)['questions']
    # print(f"len ques: {len(ques)}")
    # qid2ques = {que['question_id']: que for que in ques}
    # # combine
    # for ann in anns:
    #     que = qid2ques[ann['question_id']]
    #     ann['question'] = que['question']
    #     ann['image_url'] = que['image_url']
    # save_json(
    #     anns,
    #     os.path.join(target_dir, f"v1_avqa_{mode}_anns.json")
    # )


def process():
    from utils import save_json
    import random
    # train = load_json(f"data/vqav2/train.json")
    r1_val = load_json(
        f"data/avqa/anns/v1_avqa_r1_val_targets.json")
    r2_val = load_json(
        f"data/avqa/anns/v1_avqa_r2_val_targets.json")
    r3_val = load_json(
        f"data/avqa/anns/v1_avqa_r3_val_targets.json")
    val_verified = r1_val + r2_val + r3_val
    for ann in val_verified:
        ann['img_id'] = ann['image_id']
        ann['sent'] = ann['question']
        ann.pop('image_id')
        ann.pop('question')
    random.shuffle(val_verified)
    offset = int(len(val_verified) * 0.12)
    minival = val_verified[0:offset]
    nominival = val_verified[offset:]
    save_json(minival, f"data/avqa/minival_verified.json")
    print(f"len minival: {len(minival):,}")
    save_json(nominival, f"data/avqa/nominival_verified.json")
    print(f"len nominival: {len(nominival):,}")
    
    # r1_train = load_json(
    #     f"data/avqa/anns/v1_avqa_r1_train_targets.json")
    # r2_train = load_json(
    #     f"data/avqa/anns/v1_avqa_r2_train_targets.json")
    # r3_train = load_json(
    #     f"data/avqa/anns/v1_avqa_r3_train_targets.json")
    # train_verified = r1_train + r2_train + r3_train
    # for ann in train_verified:
    #     ann['img_id'] = ann['image_id']
    #     ann['sent'] = ann['question']
    #     ann.pop('image_id')
    #     ann.pop('question')
    # save_json(train_verified, f"data/avqa/train_verified.json")
    # print(f"len verified: {len(train_verified):,}")
    #
    # r1_train_unverified = load_json(
    #     f"data/avqa/anns/v1_avqa_r1_train_unverified_targets.json")
    # r2_train_unverified = load_json(
    #     f"data/avqa/anns/v1_avqa_r2_train_unverified_targets.json")
    # r3_train_unverified = load_json(
    #     f"data/avqa/anns/v1_avqa_r3_train_unverified_targets.json")
    # train_unverified = r1_train_unverified \
    #                    + r2_train_unverified + r3_train_unverified
    # for ann in train_unverified:
    #     ann['img_id'] = ann['image_id']
    #     ann['sent'] = ann['question']
    #     ann.pop('image_id')
    #     ann.pop('question')
    # train_all = train_verified + train_unverified
    # save_json(train_all, f"data/avqa/train_all.json")
    # print(f"len all: {len(train_all):,}")


def combine_h5():
    h5_data = h5py.File(
        f'/home/kaka/Data/vqa2/imgfeat/vqav2_trainval_avqa_train_obj36.h5', "w")
    
    # train_vqav2_feats = h5py.File(
    #     '/home/kaka/Data/vqa2/imgfeat/reserve/train_obj36.h5', 'r'
    # )
    val_vqav2_feats = h5py.File(
        '/home/kaka/Data/vqa2/imgfeat/reserve/val_obj36.h5', 'r'
    )
    # test_vqav2_feats = h5py.File(
    #     '/home/kaka/Data/vqa2/imgfeat/test_obj36.h5', 'r'
    # )train_avqa_feats[img_id]
    
    # pbar = tqdm(total=len(train_vqav2_feats))
    # for img_id in train_vqav2_feats:
    #     img_info = train_vqav2_feats[img_id]
    #     feats = img_info['features'][:].copy()
    #
    #     boxes = img_info['boxes'][:].copy()
    #     img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
    #     boxes = boxes.copy()
    #     boxes[:, (0, 2)] /= img_w
    #     boxes[:, (1, 3)] /= img_h
    #     pos_feat = np.zeros((36, 6), dtype=np.float32)
    #     pos_feat[:, :4] = boxes
    #     pos_feat[:, 4] = boxes[:, 2] - boxes[:, 0]
    #     pos_feat[:, 5] = boxes[:, 3] - boxes[:, 1]
    #
    #     img_id = f"COCO_train2014_{img_id.zfill(12)}"
    #     h5_temp = h5_data.create_group(f'{img_id}')
    #     h5_temp.create_dataset(
    #         name='boxes',
    #         data=pos_feat,
    #         dtype=np.float32
    #     )
    #     h5_temp.create_dataset(
    #         name='features',
    #         data=feats,
    #         dtype=np.float32
    #     )
    #     pbar.update(1)
    # pbar.close()
    
    pbar = tqdm(total=len(val_vqav2_feats))
    for img_id in val_vqav2_feats:
        img_info = val_vqav2_feats[img_id]
        feats = img_info['features'][:].copy()
        
        boxes = img_info['boxes'][:].copy()
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        pos_feat = np.zeros((36, 6), dtype=np.float32)
        pos_feat[:, :4] = boxes
        pos_feat[:, 4] = boxes[:, 2] - boxes[:, 0]
        pos_feat[:, 5] = boxes[:, 3] - boxes[:, 1]
        
        img_id = f"COCO_val2014_{img_id.zfill(12)}"
        h5_temp = h5_data.create_group(f'{img_id}')
        h5_temp.create_dataset(
            name='boxes',
            data=pos_feat,
            dtype=np.float32
        )
        h5_temp.create_dataset(
            name='features',
            data=feats,
            dtype=np.float32
        )
        pbar.update(1)
    pbar.close()
    
    train_avqa_feats = h5py.File(
        'data/avqa/imgfeat/avqa_train_obj36.h5', 'r'
    )
    pbar = tqdm(total=len(train_avqa_feats))
    for img_id in train_avqa_feats:
        h5_temp = h5_data.create_group(f'{img_id}')
        
        feats = train_avqa_feats[img_id]['features'][:].copy()[:36, :]
        # [x1/W, y1/H, x2/W, y2/H, , x2-x1/W, y2-y1/H]
        boxes = train_avqa_feats[img_id]['boxes'][:].copy()[:36]
        
        h5_temp.create_dataset(
            name='boxes',
            data=boxes,
            dtype=np.float32
        )
        h5_temp.create_dataset(
            name='features',
            data=feats,
            dtype=np.float32
        )
        pbar.update(1)
    pbar.close()


def np2h5():
    feat_dir = '/media/kaka/HD2T/Dataset/avqa/imgfeat/'
    h5_data = h5py.File(f'{feat_dir}/test_obj100.h5', "w")
    
    # train_vqav2_ann = load_json(f"data/vqav2/minival.json")\
    #                   + load_json(f"data/vqav2/nominival.json")
    #
    # train_vqav2_feats = h5py.File(
    #     '/home/kaka/Data/vqa2/imgfeat/reserve/val_obj36.h5', 'r'
    # )
    # train_vqav2_imgids = [a['img_id'] for a in train_vqav2_ann]
    # train_vqav2_imgids = list(set(train_vqav2_imgids))
    #
    # pbar = tqdm(total=len(train_vqav2_imgids))
    # for img_id in train_vqav2_imgids:
    #     h5_temp = h5_data.create_group(f'{img_id}')
    #
    #     img_info = train_vqav2_feats[f"{int(img_id.split('_')[-1])}"]
    #     feats = img_info['features'][:].copy()
    #
    #     boxes = img_info['boxes'][:].copy()
    #     # Normalize the boxes (to 0 ~ 1)
    #     img_h, img_w = img_info['img_hw'][:][0], img_info['img_hw'][:][1]
    #     boxes = boxes.copy()
    #     boxes[:, (0, 2)] /= img_w
    #     boxes[:, (1, 3)] /= img_h
    #     pos_feat = np.zeros((36, 6), dtype=np.float32)
    #     pos_feat[:, :4] = boxes
    #     pos_feat[:, 4] = boxes[:, 2] - boxes[:, 0]
    #     pos_feat[:, 5] = boxes[:, 3] - boxes[:, 1]
    #
    #     h5_temp.create_dataset(
    #         name='boxes',
    #         data=pos_feat,
    #         dtype=np.float32
    #     )
    #     h5_temp.create_dataset(
    #         name='features',
    #         data=feats,
    #         dtype=np.float32
    #     )
    #     pbar.update(1)
    # pbar.close()
    
    # train_anns = load_json(f"data/avqa/minival.json") \
    #              + load_json(f"data/avqa/nominival.json")
    train_anns = load_json(f"data/avqa/test.json")
    
    img_ids = [a['img_id'] for a in train_anns]
    img_ids = list(set(img_ids))
    
    pbar = tqdm(total=len(img_ids))
    for img_id in img_ids:
        h5_temp = h5_data.create_group(f'{img_id}')
        # Get image info
        # img_info.files: ['norm_bb', 'features', 'conf', 'soft_labels']
        img_info = np.load(os.path.join(feat_dir, f"test/{img_id}.npz"))
        # obj_num = img_info['num_boxes']
        feats = img_info['features'][:].copy()
        # [x1/W, y1/H, x2/W, y2/H, x2-x1/W, y2-y1/H]
        boxes = img_info['norm_bb'][:].copy()
        objects_conf = img_info['conf'][:].copy()
        
        h5_temp.create_dataset(
            name='boxes',
            data=boxes,
            dtype=np.float32
        )
        h5_temp.create_dataset(
            name='features',
            data=feats,
            dtype=np.float32
        )
        h5_temp.create_dataset(
            name='objects_conf',
            data=objects_conf,
            dtype=np.float32
        )
        pbar.update(1)
    pbar.close()


def derive_object_class():
    import torch
    feat_dir = '/media/kaka/HD2T/Dataset/avqa/imgfeat/train/'
    train_anns = load_json(f"data/avqa/ann_prompt/train_all_v3.json")
    
    obj_label = read_txt(
        f'data/vqav2/imgfeat/objects_vocab.txt')
    id2obj = {i: obj_label[i] for i in range(len(obj_label))}
    
    pbar = tqdm(total=len(train_anns))
    for ann in train_anns:
        img_id = ann['img_id']
        q_words = ann['sent'].rstrip('?').split(' ')
        
        img_info = np.load(os.path.join(feat_dir, f"{img_id}.npz"))
        # soft_conf = img_info['conf'][:].copy()
        soft_label = torch.from_numpy(
            img_info['soft_labels'][:].copy()[:36, 1:]).float()
        soft_label = soft_label.max(-1)[1]
        soft_label = soft_label.tolist()
        masked_obj_class = [id2obj[i] for i in soft_label
                            if id2obj[i] in q_words]
        masked_obj_id = [i for i in range(len(soft_label))
                         if id2obj[soft_label[i]] in q_words]
        # masked_obj_class = list(set(masked_obj_class))
        ann["masked_obj_class"] = masked_obj_class
        ann["masked_obj_id"] = masked_obj_id
        
        pbar.update(1)
    pbar.close()
    save_json(train_anns, "data/avqa/ann_prompt/train_all_v4.json")
    
    # anns = load_json("data/avqa/ann_prompt/train_all_v2.json")
    # counter_masked = 0
    # for ann in anns:
    #     masked_obj_class = ann["masked_obj_class"]
    #     if masked_obj_class:
    #         counter_masked += 1
    #         masked_obj_class = list(set(masked_obj_class))
    #         ann["masked_obj_class"] = masked_obj_class
    # print(f"len masked_obj_class: {counter_masked}: {len(anns)}")
    # save_json(anns, "data/avqa/ann_prompt/train_all_v2.json")


def combine_json():
    ann_v2 = load_json("data/advqa/ann_prompt/val_v1.json")
    for a2 in ann_v2:
        if a2['answer_type'] == 'number':
            a2['prompt'] = 'the number is'
        elif a2['answer_type'] == 'yes/no':
            a2['prompt'] = 'yes or no?'
        else:
            a2['prompt'] = 'answer is'
    save_json(ann_v2, "data/advqa/ann_prompt/val_v3.json")
    
    # ann_v2 = load_json("data/avqa/ann_prompt/nominival_v1.json")
    # for a2 in ann_v2:
    #     if a2['answer_type'] == 'number':
    #         a2['prompt'] = 'the number is'
    #     elif a2['answer_type'] == 'yes/no':
    #         a2['prompt'] = 'yes or no?'
    #     else:
    #         a2['prompt'] = 'answer is'
    # save_json(ann_v2, "data/avqa/ann_prompt/nominival_v3.json")
    
    # ann_v1 = load_json("data/avqa/ann_prompt/train_all_v1.json")
    # ann_v2 = load_json("data/avqa/ann_prompt/train_all_v2.json")
    # assert len(ann_v1) == len(ann_v2)
    #
    # for a2 in ann_v2:
    #     if a2['answer_type'] == 'number':
    #         a2['prompt'] = 'the number is'
    #     elif a2['answer_type'] == 'yes/no':
    #         a2['prompt'] = 'yes or no?'
    #     else:
    #         a2['prompt'] = 'answer is'
    # save_json(ann_v2, "data/avqa/ann_prompt/train_all_v3.json")


if __name__ == '__main__':
    # np2h5()
    # main()
    # format_annotation()
    # process()
    # vqav2_val_ann = load_json(f"data/vqav2/minival.json")\
    #                   + load_json(f"data/vqav2/nominival.json")
    # save_json(
    #     vqav2_val_ann,
    #     f"data/vqav2/val.json"
    # )
    # combine_h5()
    # derive_object_class()
    # combine_json()
    # source_dir = "/media/kaka/HD2T/Dataset/avqa/anns/raw/"
    # que_file = os.path.join(
    #     source_dir, f"v1_avqa_r1_test_questions.json")
    # ques = load_json(que_file)['questions']
    #
    # que_file = os.path.join(
    #     source_dir, f"v1_avqa_r2_test_questions.json")
    # ques += load_json(que_file)['questions']
    #
    # que_file = os.path.join(
    #     source_dir, f"v1_avqa_r3_test_questions.json")
    # ques += load_json(que_file)['questions']
    #
    # for que in ques:
    #     que['sent'] = que['question']
    #     que['img_id'] = que['image_id']
    #     que.pop('question')
    #     que.pop('image_id')
    #
    # save_json(ques, os.path.join("data/avqa", f"test.json"))
    val_ann = load_json(f"data/avqa/minival.json") + load_json(
        f"data/avqa/nominival.json")
    print(len(val_ann))
    save_json(val_ann, "data/avqa/val.json")

