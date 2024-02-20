# !/bin/bash

port_number=50002
pretrained_model_name_or_path="../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors"
obj_name='bagel'
trigger_word='bagel'
bench_mark='MVTec3D-AD'
sub_folder="test"
folder_name="test"
output_dir="../../result/${bench_mark}/${obj_name}/${sub_folder}/${folder_name}"



accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../train_multi.py \
 --log_with wandb \
 --output_dir ${output_dir} \
 --pretrained_model_name_or_path "${pretrained_model_name_or_path}" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" --beta_scale_factor 0.8 \
 --anomal_only_on_object \
 --bgrm_test \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" \
 --train_unet --train_text_encoder --d_dim 320 --latent_res 64 \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2',]" \
 --start_epoch 0 --max_train_epochs 30 \
 --do_nomal_sample \
 --do_attn_loss --do_normalized_score \
 --do_map_loss