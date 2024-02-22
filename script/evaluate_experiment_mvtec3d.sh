#! /bin/bash

bench_mark="MVTec3D-AD"
class_name="bagel"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"

sub_folder="sub_3_background_masked_sample_anomal_sample_up_16_32_64"
folder_name="attn_loss_original_normalized_score_map_loss"
output_dir="metrics"

python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}/${sub_folder}/${folder_name}/reconstruction" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3