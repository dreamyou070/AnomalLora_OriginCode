import torch
import numpy as np
from PIL import Image
import cv2
import einops
global_queries = torch.randn((64,64,1280))
global_query = einops.rearrange(global_queries, 'h w c -> (h w) c')
anomal_position = torch.zeros(4096)
normal_position = 1 - anomal_position
normal_index = torch.index_select(normal_position,1)
#normal_position = normal_position.unsqueeze(1).repeat(1, 1280) # [4096, 1280]


#object_query = global_query * normal_position # anomal query

#print(object_query.shape)