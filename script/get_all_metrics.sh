#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="leather"
layer_folder="layer_3"
sub_folder="sub_3_up_16_32_64"
folder_name="sigma_max_60_min_sigma_25_max_perlin_scale_6_max_beta_scale_0.8_min_beta_scale_0.5_back_perlin"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}"