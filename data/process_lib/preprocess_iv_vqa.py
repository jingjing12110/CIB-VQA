# @File  :preprocess_iv_vqa.py
# @Time  :2021/5/6
# @Desc  :
import os
from tqdm import tqdm

from utils import load_json, save_json
from data.process_lib.annotation_process import compute_test_target

source_dir = '/home/kaka/Data/vqa2/iv_vqa/testing/'
target_dir = 'data/iv_vqa/'
vqa2_dir = '/home/kaka/Data/vqa2/vqav2/raw/'


def format_edited_ann():
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
        dataset='iv_vqa'
    )
    save_json(
        target,
        os.path.join(target_dir, f'edited_targets.json')
    )
    

def format_original_ann():
    target = load_json(os.path.join(target_dir, f'edited_targets.json'))
    # original dataset annotation
    # anns = load_json(os.path.join(
    #     source_dir, f'original/v2_mscoco_val2014_annotations.json'))[
    #     'annotations']
    # ques = load_json(os.path.join(
    #     source_dir,
    #     f'original/v2_OpenEnded_mscoco_val2014_questions.json'))['questions']
    # print(f'len original ann: {len(anns)}')
    #
    # ans2label = load_json('data/vqav2/trainval_ans2label.json')
    #
    # for a, q in zip(anns, ques):
    #     assert a['question_id'] == q['question_id']
    #     a['question'] = q['question']
    # target = compute_test_target(
    #     anns,
    #     ans2label=ans2label,
    #     dataset='iv_vqa'
    # )
    # save_json(
    #     target,
    #     os.path.join(target_dir, f'original_targets.json')
    # )
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


def rewrite_ann():
    ann_nominival = load_json(f'data/vqav2/nominival.json')
    ann_minival = load_json(f'data/vqav2/minival.json')  # 25,994
    val_ann = ann_nominival + ann_minival
    print(f'len val_ann: {len(val_ann)}')  # 214,354
    
    img2datum = {}
    for ann in val_ann:
        img_id = ann['img_id']
        if img_id in img2datum.keys():
            img2datum[img_id].append(ann)
        else:
            img2datum[img_id] = [ann]
    print(f'img2datum len: {len(img2datum)}')
    
    # anns_edited = load_json(os.path.join(
    #     source_dir,
    #     f'edited/v2_mscoco_val2014_annotations.json'))['annotations']
    # ques_edited = load_json(os.path.join(
    #     source_dir,
    #     f'edited/v2_OpenEnded_mscoco_val2014_questions.json'))['questions']
    anns_edited = load_json(f'data/iv_vqa/edited_targets.json')
    print(f'len edited ann: {len(anns_edited)}')
    
    # img_org_ids = [a['image_id'].split('_')[0] for a in anns_edited]
    # img_org_ids = list(set(img_org_ids))
    # print(f'img_org_ids len: {len(img_org_ids)}')
    # org_anns = {}
    # for key in img2datum.keys():
    #     if key.split('_')[-1] in img_org_ids:
    #         org_anns[key] = img2datum[key]
    # print(f'org_anns len: {len(org_anns)}')
    
    edited_img2datum = {}
    for ann in anns_edited:
        img_id = ann['image_id']
        if img_id in edited_img2datum.keys():
            edited_img2datum[img_id].append(ann)
        else:
            edited_img2datum[img_id] = [ann]
    print(f'edited_img2datum len: {len(edited_img2datum)}')
    
    # ann_ori = load_json(os.path.join(
    #     source_dir,
    #     f'original/v2_mscoco_val2014_annotations.json'))['annotations']
    # que_ori = load_json(os.path.join(
    #     source_dir,
    #     f'original/v2_OpenEnded_mscoco_val2014_questions.json'))['questions']
    ann_ori = load_json(f'data/iv_vqa/original_targets.json')
    print(f'len original ann: {len(ann_ori)}')
    
    edit_org_pairs = []
    edit_org_qid_pairs = []
    for img_id, anns in edited_img2datum.items():
        qid2ann = {
            a['question_id'].split('-')[0]: a for a in anns
        }
        ann_org_all = img2datum[f"COCO_val2014_{img_id.split('_')[0]}"]
        qid2org_ann = {
            a['question_id']: a for a in ann_org_all
        }
        
        for qid, a in qid2ann.items():
            edit_org_pairs.append((a, qid2org_ann[int(qid)]))
            edit_org_qid_pairs.append(
                (a['question_id'], qid2org_ann[int(qid)]['question_id'])
            )

    print(f'edited2datum len: {len(edit_org_pairs)}')
    save_json(edit_org_pairs, 'data/iv_vqa/edit_org_pairs.json')
    save_json(edit_org_qid_pairs, 'data/iv_vqa/edit_org_qid_pairs.json')


def process_img():
    import shutil
    # # 全部是val原始文件
    # data_original = load_json(f'data/iv_vqa/original_targets.json')
    #
    # ann_nominival = load_json(f'data/vqav2/nominival.json')
    # ann_minival = load_json(f'data/vqav2/minival.json')  # 25,994
    # val_ann = ann_nominival + ann_minival
    
    data_edited = load_json(f'data/iv_vqa/edited_targets.json')
    img_ids = [d['image_id'] for d in data_edited]
    img_ids = list(set(img_ids))
    print(f'unique image len: {len(img_ids)}')
    
    img_dir_source = '/home/kaka/Data/vqa2/iv_vqa/Images/val2014/'
    img_dir_target = '/home/kaka/Data/vqa2/iv_vqa/testing/edited_images/'
    for img_id in tqdm(img_ids, ncols=80):
        shutil.copyfile(
            os.path.join(img_dir_source, f'COCO_val2014_{img_id}.jpg'),
            os.path.join(img_dir_target, f'COCO_val2014_{img_id}.jpg'),
        )


if __name__ == '__main__':
    format_edited_ann()
    # format_original_ann()
    # rewrite_ann()
    # process_img()
