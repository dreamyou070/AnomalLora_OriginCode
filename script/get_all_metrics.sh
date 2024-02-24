#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="cookie"
layer_folder="layer_3"
sub_folder="sub_3_up_16_32_64"
folder_name="back_noise_use_gaussian_timestep_600"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}"