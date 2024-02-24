#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="cookie"
layer_folder="layer_3"
sub_folder="up_16_32_64"
folder_name="attn_loss_original_normalized_score_map_loss_dist_loss_on_object_normalize_task_loss"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}"