# CIB-LXMERT 

Pytorch code (lite version) of Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering 
[[Paper](https://arxiv.org/pdf/2209.06954.pdf)] [[Slide](https://drive.google.com/file/d/12p1Pi9eWrlm3n57zQoZcps1IVqlT025V/view?usp=sharing)]. 

This repository is based on [LXMERT](https://github.com/airsplay/lxmert). 


## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Please see [data/README.md](data/README.md) for generation or download from [here](https://drive.google.com/file/d/1T7L33SBiT7ctp9qgseNJSMOomWWqVJBP/view?usp=sharing).

```angular2html
├── data
│   ├── cv_vqa
│   │   ├── edited_targets.json
│   │   └── original_targets.json
│   ├── iv_vqa
│   │   ├── edited_targets.json
│   │   └── original_targets.json
│   ├── lxmert
│   │   └── all_ans.json
│   ├── vqa_ce
│   │   ├── all_targets.json
│   │   ├── counterexample_targets.json
│   │   ├── easy_targets.json
│   │   └── hard_targets.json
│   ├── vqa_p2
│   │   ├── vqa_p2_original_targets.json
│   │   └── vqa_p2_targets.json
│   ├── vqa_rep
│   │   └── val2014_humans_vqa-rephrasings_targets.json
│   └── vqav2
│       ├── img_id_wh
│       │   ├── test.json
│       │   ├── train.json
│       │   ├── trainval.json
│       │   └── val.json
│       ├── minival.json
│       ├── nominival.json
│       ├── test.json
│       ├── train.json
│       ├── trainval_ans2label.json
│       └── trainval_label2ans.json
```

## Method 

```bash
bash train.sh 
```


## Citation

If you find our work useful in your research, please consider citing:

```tex
@article{jiang2022finetuning,
  title={Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering},
  author={Jiang, Jingjing and Liu, Ziyi and Zheng, Nanning},
  journal={arXiv preprint arXiv:2209.06954},
  year={2022}
}
```


