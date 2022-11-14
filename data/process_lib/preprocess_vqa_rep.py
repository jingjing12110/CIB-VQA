# @File  :preprocess_vqa_rep.py
# @Time  :2021/4/15
# @Desc  :
import os
from utils import load_json, save_json
from data.process_lib.annotation_process import compute_test_target

source_dir = '/home/kaka/Data/vqa2/vqa_rep/'
target_dir = 'data/vqa_rep/'
vqa2_dir = '/home/kaka/Data/vqa2/vqav2/raw/'


def rewrite_annotation():
    # val_ann = load_json(f'data/vqav2/nominival.json')
    # val_min = load_json(f'data/vqav2/minival.json')
    # val_ann = val_ann + val_min
    #
    # val_ann = {
    #     datum['question_id']: datum
    #     for datum in val_ann
    # }
    
    rephrase_ann_file = 'raw/v2_mscoco_valrep2014_humans_og_annotations.json'

    rephrase_anns = load_json(os.path.join(
        source_dir, rephrase_ann_file))['annotations']
    
    rephrase_que_file = \
        'raw/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json'
    rephrase_ques = load_json(
        os.path.join(source_dir, rephrase_que_file))['questions']
    # rephrase_ques = load_json(os.path.join(
    #     source_dir, 'val2014_humans_vqa-rephrasings_questions.json'))
    
    print(f'original len of rephrase_anns: {len(rephrase_anns)}')
    
    original_anns = []
    reph_anns = []
    combine_anns = []
    for ann, que in zip(rephrase_anns, rephrase_ques):
        assert ann['question_id'] == que['question_id']
        
        question = que['question'].strip()
        if question.endswith('/'):
            question = f"{question.rstrip('/')}?"
        
        if question == '':
            continue
        if question == '?':
            print(question)
            continue

        ann['question'] = question
        if 'rephrasing_of' in que.keys():
            ann['rephrasing_of'] = que['rephrasing_of']
            reph_anns.append(ann)
            combine_anns.append(ann)
        else:
            original_anns.append(ann)
            combine_anns.append(ann)
    print(f'original len: {len(original_anns)}\n'
          f'rephrase len: {len(reph_anns)}\n'
          f'combine len: {len(combine_anns)}')
    save_json(original_anns, os.path.join(
        source_dir, f'val2014_humans_original_annotations.json'))
    save_json(reph_anns, os.path.join(
        source_dir, f'val2014_humans_rephrase_annotations.json'))
    save_json(combine_anns, os.path.join(
        source_dir, 'val2014_humans_vqa-rephrasings_annotations.json'))
    

def format_annotation():
    # ann_mode = 'rephrase'
    # ann_mode = 'original'
    ann_mode = 'vqa-rephrasings'

    anns = load_json(os.path.join(
        source_dir, f'val2014_humans_{ann_mode}_annotations.json'))
    print(f'len of rephrase_anns: {len(anns)}')

    ans2label = load_json('data/vqav2/trainval_ans2label.json')
    
    # compute target
    target = compute_test_target(
        anns,
        ans2label=ans2label,
        dataset='vqa_rep'
    )
    save_json(target, os.path.join(
        target_dir, f'val2014_humans_{ann_mode}_targets.json'))
    

if __name__ == '__main__':
    # rewrite_annotation()
    format_annotation()
