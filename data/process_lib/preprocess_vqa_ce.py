# @File :preprocess_vqa_ce.py
# @Desc :https://github.com/cdancette/detect-shortcuts
import os
from tqdm import tqdm

from utils import load_json, save_json
from data.process_lib.annotation_process import compute_target


source_dir = '/home/kaka/Data/vqa2/vqa_ce/'
target_dir = 'data/vqa_ce/'
vqa2_dir = '/home/kaka/Data/vqa2/vqav2/raw/'


def rewrite_ann():
    ann_nominival = load_json(f'data/vqav2/nominival.json')
    ann_minival = load_json(f'data/vqav2/minival.json')  # 25,994
    val_ann = ann_nominival + ann_minival
    
    print(f'len val_ann: {len(val_ann)}')  # 214,354
    
    counterexample_qid = load_json(
        os.path.join(source_dir, f'counterexamples.json'))  # 63,298
    hard_qid = load_json(os.path.join(source_dir, f'hard.json'))  # 3,375

    counterexample, hard, easy = [], [], []
    tbar = tqdm(total=len(val_ann), ascii=True, ncols=80)
    for ann in val_ann:
        qid = ann['question_id']
        ann['img_id'] = int(ann['img_id'].split('_')[-1])
        ann['question'] = ann['sent']
        ann.pop('sent')
        if qid in counterexample_qid:
            counterexample.append(ann)
        elif qid in hard_qid:
            hard.append(ann)
        else:
            easy.append(ann)
        tbar.update(1)
    tbar.close()
    print(f'len counterexample: {len(counterexample)}\n'
          f'len hard: {len(hard)}\n'
          f'len easy: {len(easy)}\n'
          f'len 3+: {len(counterexample + hard + easy)}')
    save_json(
        counterexample,
        os.path.join(target_dir, f'counterexample_targets.json')
    )
    save_json(
        hard,
        os.path.join(target_dir, f'hard_targets.json')
    )
    save_json(
        easy,
        os.path.join(target_dir, f'easy_targets.json')
    )
    save_json(
        counterexample + hard + easy,
        os.path.join(target_dir, f'all_targets.json')
    )


if __name__ == '__main__':
    import numpy as np
    
    # rewrite_ann()
    all_targets = load_json(os.path.join(target_dir, f'all_targets.json'))
    ques_len = [len(a['question'].split(' ')) for a in all_targets]
    print(np.array(ques_len).mean())
