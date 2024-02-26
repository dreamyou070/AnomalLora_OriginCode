#! /bin/bash

bench_mark="MVTec"
class_name="wood"
layer_folder="layer_3"
sub_folder="up_16_32_64"
folder_name="sigma_max_60_min_sigma_25_max_perlin_scale_6_max_beta_scale_0.7_min_beta_scale_0.4_not_rot_not_backnoise"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}"