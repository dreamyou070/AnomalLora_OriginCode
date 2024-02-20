#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="carrot"
second_folder_name="sub_5_normal_sample_anomal_sample"
folder="attn_loss_normalized_score_map_loss"



python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${second_folder_name}/${folder}"