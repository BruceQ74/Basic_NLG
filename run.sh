#!/bin/bash

python train.py --dataset WN18RR --data_dir ../Dataset/WN18RR/text --learning_rate 1e-5 --batch_size 16  --num_epoch 1 --warmup_proportion 0.06 --encoder transformer  --bert_model ../../../model/bert_base_uncased