import torch

resized_attn_scores = [torch.randn(8, 4096, 77), torch.randn(8, 4096, 77),torch.randn(8, 4096, 77)]
concat_attn_score = torch.cat(resized_attn_scores, dim=0)  # 8, 4096, sen_len ***
    #self.resized_attn_scores = []
    #return concat_attn_score[:, :, :2]
print(concat_attn_score.shape)