#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="dowel"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
layer_folder="layer_3"
sub_folder="sub_3_up_16_32_64"
folder_name="zero_timestep_sigma_max_60_min_sigma_25_max_perlin_scale_6"
output_dir="metrics"




python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}/reconstruction" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3
