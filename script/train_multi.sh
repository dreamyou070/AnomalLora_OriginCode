# !/bin/bash
port_number=50117
pretrained_model_name_or_path="../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors"
obj_name='dowel'
trigger_word='dowel'
bench_mark='MVTec3D-AD'

layer_folder="layer_3"
sub_folder="up_16_32_64"
folder_name="zero_timestep_sigma_max_60_min_sigma_25_max_perlin_scale_6_test_noise_predicting_task_loss_do_cls_train_dist_loss"
output_dir="../../result/${bench_mark}/${obj_name}/${layer_folder}/${sub_folder}/${folder_name}"
# --use_noise_scheduler --min_timestep 399 --max_timestep 400 \
# --use_text_time_embedding
# --do_dist_loss --mahalanobis_only_object --mahalanobis_normalize --dist_loss_with_max \
#

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_3_4_config \
 --main_process_port $port_number ../train_multi.py \
 --log_with wandb \
 --output_dir ${output_dir} \
 --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --beta_scale_factor 1.0 \
 --anomal_source_path "../../../MyData/anomal_source" \
 --anomal_only_on_object \
 --anomal_p 0.04 \
 --bgrm_test \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" \
 --train_unet --train_text_encoder --d_dim 320 --latent_res 64 \
 --network_dim 64 --network_alpha 4 \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_1_attentions_2_transformer_blocks_0_attn2',]" \
 --start_epoch 0 --max_train_epochs 30 \
 --do_anomal_sample --do_background_masked_sample \
 --do_dist_loss --mahalanobis_only_object --mahalanobis_normalize --dist_loss_with_max \
 --do_attn_loss --do_cls_train \
 --do_map_loss \
 --test_noise_predicting_task_loss \
 --back_noise_use_gaussian --max_sigma 100 --min_sigma 30 --max_perlin_scale 4 \
 --max_beta_scale 1 --min_beta_scale 0