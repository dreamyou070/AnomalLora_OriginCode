#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="cookie"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
layer_folder="layer_2_16_64"
sub_folder="sub_3_up_16_2_64_2"
folder_name="back_noise_use_perlin_zero_timestep"
output_dir="metrics"


python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}/reconstruction_timestep_0" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3
