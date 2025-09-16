import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnPairHead(nn.Module):
    def __init__(self, dim=64, heads=4, ffn=128, pdrop=0.1):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)

        self.mha_st = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                            dropout=pdrop, batch_first=True)
        self.mha_ts = nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                            dropout=pdrop, batch_first=True)

        self.ffn_s  = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, ffn), nn.GELU(),
            nn.Dropout(pdrop), nn.Linear(ffn, dim))
        self.ffn_t  = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, ffn), nn.GELU(),
            nn.Dropout(pdrop), nn.Linear(ffn, dim))

        fuse_in = dim * 4
        self.cls = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, 256), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(256, 1)
        )
        self.tau = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _key_padding_mask(lengths, Lmax):
        device = lengths.device
        rng = torch.arange(Lmax, device=device)[None, :]
        return rng >= lengths[:, None]

    @staticmethod
    def _masked_mean(x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(-1)
        s = (x * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp_min(1.0)
        return s / d

    def forward(self, f_s, f_t, len_s, len_t):
        B, Ls, C = f_s.shape
        Lt = f_t.shape[1]

        s = self.norm_s(f_s)
        t = self.norm_t(f_t)

        kp_t = self._key_padding_mask(len_t, Lt)
        kp_s = self._key_padding_mask(len_s, Ls)

        s_att, _ = self.mha_st(query=s, key=t, value=t, key_padding_mask=kp_t)
        s = s + s_att
        s = s + self.ffn_s(s)

        t_att, _ = self.mha_ts(query=t, key=s, value=s, key_padding_mask=kp_s)
        t = t + t_att
        t = t + self.ffn_t(t)

        s_pool = self._masked_mean(s, len_s)
        t_pool = self._masked_mean(t, len_t)

        fuse = torch.cat([s_pool, t_pool, (s_pool - t_pool).abs(), s_pool * t_pool], dim=-1)  # B×(4C)
        logit = self.cls(fuse) / self.tau.clamp_min(0.1)
        return logit
