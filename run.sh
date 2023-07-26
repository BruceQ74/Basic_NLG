#!/bin/bash

python train.py --dataset couplets --data_dir ./data/couplets --learning_rate 1e-5 --batch_size 8  --num_epoch 30 --warmup_proportion 0.06 --bert_model bert-base-chinese