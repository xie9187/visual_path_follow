#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python script/training_rdpg_BPTT.py --model_dir /home/linhai/Work/catkin_ws/data/vpf_data/saved_network --buffer_size 1000 --max_training_step 5000000
