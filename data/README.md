# Data Processing

## Training Data: VQA v2

- Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset) from https://nlp.cs.unc.edu/data/lxmert_data/. 
- Converting .tsv to .h5 (process_lib/tsv2h5.py)

## Testing Data

- [VQA-Rephrasings](https://facebookresearch.github.io/VQA-Rephrasings/): ```process_lib/preprocess_vqa_rep.py```  
    [original + rephrase]
- [VQA P2](https://github.com/SpencerWhitehead/vqap2): ```process_lib/preprocess_vqa_p2.py```  
    [original + paraphrastic/rephrase + synonymous + antonymous]
- [VQA-CE](https://github.com/cdancette/detect-shortcuts):  ```process_lib/preprocess_vqa_ce.py```  
    [overall + counterexample + easy]  
- [IV-VQA, CV-VQA](https://github.com/AgarwalVedika/CausalVQA): ```process_lib/preprocess_iv_vqa.py``` and ```process_lib/preprocess_cv_vqa.py```
    [real + realNE + edit]
  - training on original VQA v2.0 train set, and testing on IV-VQA and CV-VQA testing set

