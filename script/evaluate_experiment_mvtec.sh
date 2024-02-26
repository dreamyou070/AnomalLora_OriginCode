#! /bin/bash

bench_mark="MVTec"
class_name="screw"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
layer_folder="layer_3"
sub_folder="up_16_32_64"
folder_name="test_2"

base_dir="../../result/${bench_mark}/${class_name}/${layer_folder}/${sub_folder}/${folder_name}/reconstruction" \
output_dir="metrics"

python ../evaluation/evaluation_code_MVTec/evaluate_experiment_2.py \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3