import numpy as np

# https://wandb.ai/nerdk312/DUQ/reports/120421-Spectral-values-analysis--Vmlldzo1OTk4MTI
# Examine using supervised cross entropy loss in conjunction with the instance discrimination loss to see whether the features learnt using self-supervised pretraining is more robust to outliers
# https://koreascience.kr/article/JAKO202219455120547.page
class Metric():
    def __init__(self, embed_dim, mode,  **kwargs):
        self.mode      = mode
        self.embed_dim = embed_dim
        self.requires = ['features']
        self.name     = 'rho_spectrum@'+str(mode)


    def __call__(self, features):
        from sklearn.decomposition import TruncatedSVD
        from scipy.stats import entropy
        import torch


        if isinstance(features, torch.Tensor):
            _,s,_ = torch.svd(features)  # Nawid - obtains singular values of svd
            s     = s.cpu().numpy()
        else:
            svd = TruncatedSVD(n_components=self.embed_dim-1, n_iter=7, random_state=42)
            svd.fit(features)
            s = svd.singular_values_


        if self.mode!=0:
            s = s[np.abs(self.mode)-1:] # Nawid - select a subset of the values (from the mode -1 value to the last value)
        s_norm  = s/np.sum(s) # Nawid- normalise by sum of spectral values
        uniform = np.ones(len(s))/(len(s))


        if self.mode<0:
            kl = entropy(s_norm, uniform) # Nawid - calculate the KL divergence between a uniform distribution and the singular values
        if self.mode>0:
            kl = entropy(uniform, s_norm)
        if self.mode==0:
            kl = s_norm


        return kl
