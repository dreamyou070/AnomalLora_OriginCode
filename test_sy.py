import torch
resized_query = torch.randn(8, 64*64, 320)
query = resized_query
print(query.shape)