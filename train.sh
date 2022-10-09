#!/usr/bin/env bash

# *****************************************************************************
name=lxmert_pretrained_Epoch20LXRT_CIB_epoch10_beta1e3_lr3e5_cosine08
output=snap/final/lxmert-CIB-NewUP/$name
mkdir -p $output/src
cp -r src/baseline/lxmert/lxrt $output/src/
cp -r src/baseline/lxmert/pretrain $output/src/
cp -r src/baseline/lxmert/vqa_model.py $output/src/
cp -r train.py $output/src/
cp $0 $output/run.sh
# Training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python train.py \
    --train train --valid val \
    --numWorkers 8 --beta 1e-3 \
    --loadLXMERT "snap/pretrained/Epoch20" \
    --bs 104 --optim AdamW --lr 3e-5 --epochs 10 \
    --warmup_ratio 0.1 --lr_mode cosine \
    --tqdm --tf_writer True \
    --output $output ${@:3}
#Testing
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python train.py \
    --train train --valid "" \
    --test minival \
    --load snap/final/lxmert-CIB-NewUP/"$name"/BEST \
    --numWorkers 8 --tf_writer False \
    --bs 1024 --optim AdamW \
    --tqdm --output $output ${@:3}
# All Testing
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python train.py \
    --train train --valid ""  \
    --test test \
    --load snap/final/lxmert-CIB-NewUP/"$name"/BEST \
    --numWorkers 8 --tf_writer False \
    --bs 1024 --optim AdamW \
    --tqdm --output $output ${@:3}
# consistency evaluation
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python cs_evaluate.py \
    --load snap/final/lxmert-CIB-NewUP/"$name"/BEST \
    --bs 1024
