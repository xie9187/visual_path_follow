#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python python script/training_vpf_rl.py --model_dir /home/linhai/Work/catkin_ws/data/vpf_data/saved_network --buffer_size 1000 --max_training_step 5000000
