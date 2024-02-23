#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="cookie"
sub_folder="sub_3_background_masked_sample_anomal_sample_up_16_32_64"
#folder_name="attn_loss_original_normalized_score_map_loss_noise_predicting_task_loss"
folder_name="attn_loss_original_normalized_score_map_loss"


python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${sub_folder}/${folder_name}"