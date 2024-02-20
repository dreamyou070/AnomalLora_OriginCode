#! /bin/bash

class_name="carrot"
second_folder_name="1_3_background_masked_sample_attn_loss_dist_loss_map_loss_focal_only_zero_timestep"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name}