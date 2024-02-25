#! /bin/bash

bench_mark="MVTec"
class_name="bootle"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
layer_folder="layer_3"
sub_folder="up_16_32_64"
folder_name="sigma_max_60_min_sigma_25_max_perlin_scale_6_max_beta_scale_0.8_min_beta_scale_0.5_back_perlin_cropping_test_anomal_p_0.01_new_code"

base_dir="../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}/reconstruction_infer_with_crop" \
output_dir="metrics"

python ../evaluation/evaluation_code_MVTec/evaluate_experiment_2.py \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3