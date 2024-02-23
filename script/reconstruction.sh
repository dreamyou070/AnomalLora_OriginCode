# !/bin/bash

port_number=50101
obj_name='cookie'
caption='cookie'
sub_folder="sub_3_up_16_0_2_32_64"
folder_name="back_noise_use_gaussian_zero_timestep"
bench_mark="MVTec3D-AD"
position_embedding_layer="down_blocks_0_attentions_0_transformer_blocks_0_attn1"
# [0.7,0.75,0.8,0.85,0.9,0.95,0.98]
accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${obj_name}/${sub_folder}/${folder_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}" \
 --latent_res 64 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2',
                                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                                    'up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                                    'up_blocks_1_attentions_0_transformer_blocks_0_attn2',]" \
 --d_dim 320 --use_position_embedder --position_embedding_layer ${position_embedding_layer} \
 --threds [0.5]