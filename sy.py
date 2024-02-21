import os, torch

sample_num = 10
dim = 30
query = torch.randn(sample_num, dim)
""" normalize """
normalized_query = torch.nn.functional.normalize(query, p=2, dim=1, eps=1e-12, out=None)
norm = torch.norm(normalized_query, p=2, dim=1, keepdim=False, out=None)
print(norm)
#print(normalized_query)