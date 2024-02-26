# !/bin/bash

port_number=51012

python ../data_check.py \
 --obj_name 'pill' \
 --anomal_only_on_object \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.5 \
 --anomal_max_beta_scale 0.8 \
 --back_min_perlin_scale 0 \
 --back_max_perlin_scale 6 \
 --back_trg_beta 0.0 \
 --anomal_p 0.03