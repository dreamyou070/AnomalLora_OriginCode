# !/bin/bash

port_number=50011
obj_name='bottle'
caption='bottle'

layer_folder="layer_3"
sub_folder="up_16_32_64"
folder_name="sigma_max_60_min_sigma_25_max_perlin_scale_6_max_beta_scale_0.8_min_beta_scale_0.5_back_perlin_cropping_test_anomal_p_0.01_new_code"
bench_mark="MVTec"
position_embedding_layer="down_blocks_0_attentions_0_transformer_blocks_0_attn1"

# 'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
# --object_crop

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../reconstruction.py \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --network_dim 64 --network_alpha 4 --network_folder "../../result/${bench_mark}/${obj_name}/${layer_folder}/${sub_folder}/${folder_name}/models" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}/${obj_name}/test" \
 --obj_name "${obj_name}" --prompt "${caption}" \
 --latent_res 64 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2',
                                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                                    'up_blocks_1_attentions_2_transformer_blocks_0_attn2',]" \
 --d_dim 320 --use_position_embedder --position_embedding_layer ${position_embedding_layer} \
 --threds [0.5]