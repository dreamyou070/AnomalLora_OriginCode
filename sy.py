import torch

org = torch.randn(8,64*64,7)
trg = org[:,:,:2]
print(trg.shape)