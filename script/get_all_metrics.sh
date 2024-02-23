#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="cookie"
sub_folder="sub_3_up_16_0_2_32_64"
folder_name="back_noise_use_perlin_zero_timestep"

python ../evaluation/get_all_metrics.py \
  --class_name ${class_name} \
  --base_save_dir "../../result/${bench_mark}/${class_name}/${sub_folder}/${folder_name}"


