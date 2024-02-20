#! /bin/bash

class_name="cookie"
bench_mark="MVTec3D-AD"
dataset_dir="../../../MyData/anomaly_detection/${bench_mark}"
<<<<<<< HEAD
sub_folder="sub_3_background_masked_sample_anomal_sample"
folder="attn_loss_normalized_score_map_loss"
=======
sub_folder="1_5_anormal_sample_background_masked_sample_attn_loss_dist_loss_map_loss_focal_only_zero_timestep_normalized_score"

>>>>>>> parent of 4a75c31 (Update evaluate_experiment_mvtec3d.sh)
output_dir="metrics"


python ../evaluation/evaluation_code_MVTec3D-AD/evaluate_experiment.py \
     --base_dir "../../result/${bench_mark}/${class_name}/${sub_folder}/${folder}/reconstruction" \
     --dataset_base_dir "${dataset_dir}" \
     --anomaly_maps_dir "${base_dir}" \
     --output_dir "${output_dir}" \
     --evaluated_objects "${class_name}" \
     --pro_integration_limit 0.3