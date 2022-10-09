# @File :preprocess_advqa.py
# @Time :2021/9/23
# @Desc :
import os
from tqdm import tqdm
import numpy as np
import h5py
from tqdm import tqdm

from utils import load_json, save_json
from data.process_lib.annotation_process import preprocess_answer, get_score


source_dir = '/media/kaka/HD2T/Dataset/advqa'
target_dir = 'data/advqa'


def compute_test_target(answers_dset, ans2label, dataset='advqa'):
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
        if dataset == 'advqa':
            target.append({
                'question_id': ans_entry['question_id'],
                'sent': ans_entry['sent'],
                'img_id': ans_entry['img_id'],
                'label': label,
                'answer_type': ans_entry['answer_type']
            })
        tbar.update(1)
    tbar.close()
    print(f'len target: {len(target)}')
    return target


def combine_ann_que():
    ann_file = os.path.join(
        source_dir, f"v1_mscoco_val2017_advqa_annotations.json")
    anns = load_json(ann_file)['annotations']
    que_file = os.path.join(
        source_dir, f"v1_OpenEnded_mscoco_val2017_advqa_questions.json")
    ques = load_json(que_file)['questions']
    print(f"len anns: {len(anns)}\n "
          f"len ques: {len(ques)}")
    qid2ques = {que['question_id']: que for que in ques}
    # combine
    for ann in anns:
        que = qid2ques[ann['question_id']]
        ann['sent'] = que['question']
        ann['img_id'] = f"COCO_val2014_{str(ann['image_id']).zfill(12)}"
        ann.pop('image_id')
    
    save_json(
        anns,
        os.path.join(target_dir, f"val.json")
    )
    

def format_ann():
    anns = load_json(os.path.join(
        target_dir, f"val.json"))
    print(f'len of anns: {len(anns)}')
    
    ans2label = load_json('data/vqav2/trainval_ans2label.json')
    # compute target
    target = compute_test_target(
        anns,
        ans2label=ans2label,
        dataset='advqa'
    )
    print(f'len of target: {len(target)}')
    # save_json(target, os.path.join(
    #     target_dir, f'v1_avqa_{mode}_{verify}_targets.json'))
    save_json(target, os.path.join(
        target_dir, f'val_targets.json'))


if __name__ == '__main__':
    # combine_ann_que()
    # format_ann()
    que_file = os.path.join(
        source_dir, f"v1_OpenEnded_mscoco_testdev2015_advqa_questions.json")
    ques = load_json(que_file)['questions']
    for que in ques:
        que['sent'] = que['question']
        que['img_id'] = f"COCO_test2015_{str(que['image_id']).zfill(12)}"
        que.pop('question')
        que.pop('image_id')
        
    save_json(
        ques,
        os.path.join(target_dir, f"test.json"))
