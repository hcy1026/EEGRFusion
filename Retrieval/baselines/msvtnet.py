import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# 你把这个 import 路径改成你项目里 MSVTNet 的实际位置
# 例如：from braindecode.models.msvtnet import MSVTNet as _MSVTNet
from .modules.msvtnetmodule import MSVTNet as _MSVTNet
from Retrieval.loss import ClipLoss


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class MSVTTokenEncoder(nn.Module):
    """
    用原始 _MSVTNet 生成 transformer token（不改原模型），并对齐成 [B, 63, 250]，
    以适配你现有的 Enc_eeg(PatchEmbedding) 管线。

    同时提供:
      self.encoder -> 内部 transformer 编码器（用于 count_params(eeg_model.encoder.encoder)）
    """
    def __init__(self, num_channels=63, sequence_length=250, d_model=250, out_dim=250):
        super().__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.out_dim = out_dim

        # 实例化原 MSVTNet（其内部分类头需要 n_outputs，但我们不会用 logits）
        self.backbone = _MSVTNet(
            n_chans=num_channels,
            n_times=sequence_length,
            n_outputs=1,  # dummy
        )

        # 让参数统计兼容你原来的写法：
        # - eeg_model.encoder            -> 这个 MSVTTokenEncoder
        # - eeg_model.encoder.encoder    -> transformer encoder(真正的“backbone(transformer)”)
        self.encoder = self.backbone.transformer.trans  # nn.TransformerEncoder

        # token 数对齐到 63：对 token-length 维做 AdaptiveAvgPool1d
        self.pool_to_63 = nn.AdaptiveAvgPool1d(63)

        # token embedding dim 对齐到 250（你给的 out_dim=250）
        d_msvt = int(self.backbone.transformer.cls_embedding.shape[-1])
        self.proj_to_250 = nn.Linear(d_msvt, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 63, 250]
        return: [B, 63, 250]  (对齐 ATMS 的 Enc_eeg 输入)
        """
        # 1) MSVT 的多尺度卷积产生 tokens: [B, T', Dm]
        x_ = self.backbone.ensure_dim(x)  # [B,1,C,T]
        x_list = [tsconv(x_) for tsconv in self.backbone.mstsconv]  # list of [B,T',Db]
        tokens = torch.cat(x_list, dim=2)  # [B, T', Dm]

        # 2) 不调用 backbone.transformer.forward()（它只返回 CLS）
        #    我们手动走一遍，拿到所有 token（含CLS）
        tr = self.backbone.transformer
        b = tokens.size(0)
        seq = torch.cat((tr.cls_embedding.expand(b, -1, -1), tokens), dim=1)  # [B,1+T',Dm]
        seq = tr.pos_embedding(seq)
        seq = tr.dropout(seq)
        seq = tr.trans(seq)  # [B,1+T',Dm]

        tok = seq[:, 1:, :]  # 去掉CLS -> [B,T',Dm]

        # 3) token-length 对齐到 63
        tok = tok.transpose(1, 2)      # [B,Dm,T']
        tok = self.pool_to_63(tok)     # [B,Dm,63]
        tok = tok.transpose(1, 2)      # [B,63,Dm]

        # 4) embedding dim 对齐到 250
        tok = self.proj_to_250(tok)    # [B,63,250]
        return tok


class MSVTNet(nn.Module):
    """
    作为 args.encoder_type = "MSVTNet" 的整模型：
      forward(eeg, subject_ids) -> [B, 1024]

    对齐你现有 ATMS 管线：
      encoder -> Enc_eeg -> Proj_eeg

    并补齐训练/统计依赖字段：
      - self.encoder (并且 self.encoder.encoder 存在)
      - self.logit_scale, self.loss_func
    """
    def __init__(self,
                 num_channels=63,
                 sequence_length=250,
                 proj_dim=1024,
                 d_model=250,
                 out_dim=250):
        super().__init__()

        # 这一步输出 [B,63,250]，给 Enc_eeg 用
        self.encoder = MSVTTokenEncoder(
            num_channels=num_channels,
            sequence_length=sequence_length,
            d_model=d_model,
            out_dim=out_dim,
        )

        # 复用你文件里已有的模块（ATMS_retrieval_metrics.py 里已经定义了）
        self.enc_eeg = Enc_eeg()                  # 输出 flatten 后默认是 1440
        self.proj_eeg = Proj_eeg(proj_dim=proj_dim)

        # 训练用：和 ATMS 保持一致
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids=None):
        # x: [B,63,250]
        x = self.encoder(x)            # [B,63,250]
        eeg_embedding = self.enc_eeg(x)  # [B,1440]
        out = self.proj_eeg(eeg_embedding)  # [B,1024]
        out = F.normalize(out, dim=-1)
        return out