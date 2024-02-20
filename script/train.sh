# !/bin/bash

port_number=50323

obj_name='carrot'
trigger_word='carrot'
bench_mark='MVTec3D-AD'
save_folder_name="do_normal_sample_do_anomal_sample_dist_loss_attn_loss_normalized_score"

accelerate launch --config_file ../../../gpu_config/gpu_0_1_2_config \
 --main_process_port $port_number ../train.py \
 --log_with wandb \
 --output_dir "../../result/${bench_mark}/${obj_name}/${save_folder_name}" \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --beta_scale_factor 0.8 \
 --bgrm_test \
 --anomal_source_path "../../../MyData/anomal_source" \
 --anomal_only_on_object \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" \
 --train_unet --train_text_encoder \
 --d_dim 320 --latent_res 64 \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \
 --start_epoch 0 --max_train_epochs 30 \
 --do_normal_sample --do_anomal_sample \
 --do_dist_loss \
 --do_attn_loss --do_normalized_score