# @File :preprocess_data.py
# @Time :2021/11/15
# @Desc :
import os
from tqdm import tqdm

from utils import load_json, save_json
from data.process_lib.annotation_process import compute_test_target


def format_edited_ann():
    source_dir = "/media/kaka/HD2T/Dataset/iv_vqa/ann"
    anns = load_json(os.path.join(
        source_dir, f'v2_mscoco_val2014_annotations.json'))['annotations']
    ques = load_json(os.path.join(
        source_dir,
        f'v2_OpenEnded_mscoco_val2014_questions.json'))['questions']
    print(f'len edited ann: {len(anns)}')
    
    ans2label = load_json('data/vqav2/trainval_ans2label.json')
    
    for a, q in zip(anns, ques):
        assert a['question_id'] == q['question_id']
        a['question'] = q['question']
    target = compute_test_target(
        anns,
        ans2label=ans2label,
        dataset='iv_vqa'
    )
    save_json(
        target,
        os.path.join(source_dir, f'val_edited_targets.json')
    )


def format_original_ann():
    source_dir = "/media/kaka/HD2T/Dataset/iv_vqa/ann"
    target = load_json(os.path.join(source_dir, f'edited_targets.json'))
    
    ann_nominival = load_json(f'data/vqav2/nominival.json')
    ann_minival = load_json(f'data/vqav2/minival.json')  # 25,994
    val_ann = ann_nominival + ann_minival
    val_ann = {a['question_id']: a for a in val_ann}

    qids = [a['question_id'].split('-')[0] for a in target]
    qids = list(set(qids))
    print(f'len of qids: {len(qids)}')
    original_anns = []
    for qid in qids:
        ann = val_ann[int(qid)]
        ann['image_id'] = ann['img_id']
        ann['question'] = ann['sent']
        ann.pop('sent')
        ann.pop('img_id')
        original_anns.append(ann)
    save_json(
        original_anns, os.path.join(source_dir, f'val_original_targets.json'))


def process_img():
    img_dir = "/media/kaka/HD2T/Dataset/iv_vqa/Image/val2014/"
    for img_name in os.listdir(img_dir):
        img_file = os.path.join(img_dir, img_name)



if __name__ == '__main__':
    process_img()
    # format_original_ann()
    # format_edited_ann()
