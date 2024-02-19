# !/bin/bash

port_number=50030

obj_name='carrot'
trigger_word='carrot'
bench_mark='MVTec3D-AD'

# no anomal source

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../collect_normal_feature.py \
 --log_with wandb \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --output_dir "../../result/${bench_mark}/${obj_name}/0_do_object_detection_different_code" \
 --network_weights "../../result/${bench_mark}/${obj_name}/0_do_object_detection_different_code/models/epoch-000004.safetensors" \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --beta_scale_factor 0.8 \
 --anomal_only_on_object \
 --trigger_word "${trigger_word}" --obj_name "${obj_name}" --train_unet --train_text_encoder --d_dim 320 --latent_res 64 \
 --position_embedding_layer 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' \
 --trg_layer_list "['up_blocks_3_attentions_2_transformer_blocks_0_attn2']" \