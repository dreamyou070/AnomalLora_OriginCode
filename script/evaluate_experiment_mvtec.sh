#! /bin/bash

class_name="bagel"
dataset_cat="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${dataset_cat}"
sub_folder="1_2_anomal_sample_attn_loss_dist_loss_map_loss_only_zero_timestep"
base_dir="../../result/${class_name}/${sub_folder}/reconstruction"

output_dir="metrics"

python ../evaluation/evaluation_code_MVTec/evaluate_experiment_2.py \
     --anomaly_maps_dir "${anomaly_maps_dir}" \
     --output_dir "${output_dir}" \
     --base_dir "${base_dir}" \
     --dataset_base_dir "${dataset_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3