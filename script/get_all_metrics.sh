#! /bin/bash

class_name="carrot"
second_folder_name="sub_3_background_masked_sample_anomal_sample"
folder="attn_loss_map_loss"
bench_mark="MVTec3D-AD"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --second_folder_name ${second_folder_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${second_folder_name}"