#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="cookie"
sub_folder="up_16_0_2_32_64"
folder_name="back_noise_use_gaussian_400_timestep"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/layer_4/${sub_folder}/${folder_name}"


