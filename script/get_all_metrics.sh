#! /bin/bash

class_name="carrot"
second_folder_name="1_2_anomal_sample_attn_loss_dist_loss_map_loss_only_zero_timestep"
bench_mark="MVTec3D-AD"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${second_folder_name}"