# @File  :preprocess_cv_vqa.py
# @Time  :2021/5/6
# @Desc  :
import os
from tqdm import tqdm

from utils import load_json, save_json
from data.process_lib.annotation_process import compute_test_target

source_dir = '/home/kaka/Data/vqa2/cv_vqa/testing/'
target_dir = 'data/cv_vqa/'


def format_ann():
    anns = load_json(os.path.join(
        source_dir, f'edited/v2_mscoco_val2014_annotations.json'))['annotations']
    ques = load_json(os.path.join(
        source_dir,
        f'edited/v2_OpenEnded_mscoco_val2014_questions.json'))['questions']
    print(f'len edited ann: {len(anns)}')
    
    ans2label = load_json('data/vqav2/trainval_ans2label.json')
    
    for a, q in zip(anns, ques):
        assert a['question_id'] == q['question_id']
        a['question'] = q['question']
    target = compute_test_target(
        anns,
        ans2label=ans2label,
        dataset='cv_vqa'
    )
    save_json(
        target,
        os.path.join(target_dir, f'edited_targets.json')
    )

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
        original_anns, os.path.join(target_dir, f'original_targets.json'))


def process_img():
    import shutil

    data_edited = load_json(f'data/cv_vqa/edited_targets.json')
    
    img_ids = [d['image_id'] for d in data_edited]
    img_ids = list(set(img_ids))
    print(f'unique image len: {len(img_ids)}')
    
    img_dir_source = '/home/kaka/Data/vqa2/cv_vqa/Images/val2014/'
    img_dir_target = '/home/kaka/Data/vqa2/cv_vqa/testing/edited_images/'
    for img_id in tqdm(img_ids, ncols=80):
        shutil.copyfile(
            os.path.join(img_dir_source, f'COCO_val2014_{img_id}.jpg'),
            os.path.join(img_dir_target, f'COCO_val2014_{img_id}.jpg'),
        )


if __name__ == '__main__':
    format_ann()
    process_img()

