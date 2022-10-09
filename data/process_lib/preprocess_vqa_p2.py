# @File  :preprocess_vqa_p2.py
# @Time  :2021/5/6
# @Desc  :
import os
from tqdm import tqdm

from utils import load_json, save_json
from data.process_lib.annotation_process import compute_test_target


source_dir = '/home/kaka/Data/vqa2/vqa_p2/'
target_dir = 'data/vqa_p2/'
vqa2_dir = '/home/kaka/Data/vqa2/vqav2/raw/'


def format_ann():
    ann = load_json(os.path.join(
        source_dir, f'vqap2.annotations.json'))['annotations']
    que = load_json(os.path.join(
        source_dir, f'vqap2.questions.json'))['questions']
    print(f'len ann: {len(ann)}')

    ans2label = load_json('data/vqav2/trainval_ans2label.json')
    
    for a, q in zip(ann, que):
        assert a['question_id'] == q['question_id']
        a['question'] = q['question']
    
    # compute target
    target = compute_test_target(
        ann,
        ans2label=ans2label,
        dataset='vqa_p2'
    )
    save_json(
        target,
        os.path.join(target_dir, f'vqa_p2_targets.json')
    )
    

def find_original_ann():
    ann = load_json(os.path.join(target_dir, f'vqa_p2_targets.json'))
    img_ann = {
        f"{datum['original_id']}-{datum['image_id']}": datum
        for datum in ann
    }
    # org_qid = [a['original_id'] for a in ann]
    # find original annatation
    # print(f'len org_qid: {len(list(set(org_qid)))}')
    
    val_ann = load_json(f'data/vqav2/nominival.json')
    val_min = load_json(f'data/vqav2/minival.json')
    val_ann = val_ann + val_min
    print(f'len val_ann: {len(val_ann)}')
    img_val_ann = {
        f"{datum['question_id']}-{int(datum['img_id'].split('_')[-1])}": datum
        for datum in val_ann
    }
    
    org_anns = []
    tbar = tqdm(total=len(img_ann), ncols=80)
    for key in img_ann.keys():
        ori_ann = img_val_ann[key]
        ori_ann['image_id'] = int(ori_ann['img_id'].split('_')[-1])
        ori_ann['question'] = ori_ann['sent']
        ori_ann.pop('sent')
        ori_ann.pop('img_id')
        org_anns.append(ori_ann)

        tbar.update(1)
    tbar.close()
    print(f'len org_ann: {len(org_anns)}')
    save_json(org_anns, f'data/vqa_p2/vqa_p2_original_targets.json')


if __name__ == '__main__':
    format_ann()
    # ann = load_json(os.path.join(target_dir, f'vqap2_targets.json'))
    # print(f'len total: {len(ann)}')
    #
    # org, reph, syn, ant = [], [], [], []
    # for a in ann:
    #     perturbation = a['perturbation']
    #     if perturbation == 'para':
    #         reph.append(a)
    #     elif perturbation == 'syn':
    #         syn.append(a)
    #     elif perturbation == 'ant':
    #         ant.append(a)
    #     else:
    #         print(a)
    # print(f'len reph: {len(reph)}\n'
    #       f'len syn: {len(syn)}\n'
    #       f'len ant: {len(ant)}')
    
    # find_original_ann()
    original_ann = load_json('data/vqa_p2/vqa_p2_original_targets.json')
    print(len(original_ann))
    reph_ann = load_json('data/vqa_p2/vqa_p2_targets.json')
    print(len(reph_ann))
    
    original_ques = [a['question'] for a in original_ann]
    org_lens = [len(a.split(' ')) for a in original_ques]
    
    reph_ques = [a['question'] for a in reph_ann]
    reph_lens = [len(a.split(' ')) for a in reph_ques]

    all_ques = original_ques + reph_ques
    all_lens = [len(a.split(' ')) for a in all_ques]


