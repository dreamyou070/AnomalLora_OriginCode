#! /bin/bash

class_name="carrot"
second_folder_name="attn_loss_dist_loss_map_loss_focal_only_zero_timestep_normalized_score"
bench_mark="MVTec3D-AD"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${second_folder_name}"