#!/bin/bash

python train.py --dataset test --data_dir ./data/test --learning_rate 1e-5 --batch_size 1  --num_epoch 1 --warmup_proportion 0.06