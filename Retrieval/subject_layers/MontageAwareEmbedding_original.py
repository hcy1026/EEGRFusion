import torch
import torch.nn as nn

# 复用你现有 Embed.py 里的实现，确保行为一致
from .Embed import PositionalEmbedding, TemporalEmbedding, SubjectEmbedding, TimeFeatureEmbedding


class MontageAwareEmbedding(nn.Module):
    """
    在 DataEmbedding 的基础上加入 montage-aware：
    token = value_embedding(x) + coord_embedding(coords) (+ temporal + positional) (+ subject token)

    兼容原始 DataEmbedding 的接口：
        forward(x, x_mark, subject_ids=None, mask=None) -> [B, L(+1), d_model]
    并提供 set_coords(coords) 用于在主程序中注入电极坐标。

    预期输入：
        x:      [B, L, c_in]  (你这里通常是 [B, 63, 250]，c_in=seq_len)
        x_mark: 任意（可为 None），形状保持与原 DataEmbedding 用法一致
        coords: [L, coord_dim]，由主程序根据 ch_names 生成并 set_coords
    """
    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
        joint_train: bool = False,
        num_subjects: int = None,
        coord_dim: int = 3,
        coord_scale: float = 1.0,
        use_coords: bool = True,
    ):
        super().__init__()
        self.joint_train = joint_train
        self.use_coords = use_coords
        self.coord_scale = coord_scale

        # 与 DataEmbedding 保持一致：value embedding
        if joint_train and num_subjects is not None:
            self.value_embedding = nn.ModuleDict({
                str(subject_id): nn.Linear(c_in, d_model) for subject_id in range(num_subjects)
            })
        else:
            self.value_embedding = nn.Linear(c_in, d_model)

        # 与 DataEmbedding 保持一致：pos / temporal / dropout / subject / mask
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        self.mask_token = nn.Parameter(torch.randn(1, d_model))
        self.joint_train = joint_train

        # 新增：坐标 embedding（轻量 MLP，避免引入不必要复杂度）
        # 说明：d_model 你这里是 250，MLP 规模很小，不会明显增参
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # coords buffer：运行时 set_coords 注入
        # 先注册一个空 buffer，不影响加载/保存
        self.register_buffer("coords", torch.zeros(1, coord_dim), persistent=False)

    @torch.no_grad()
    def set_coords(self, coords: torch.Tensor):
        """
        coords: [L, coord_dim]，L 必须与你实际 token 数（63）一致
        """
        if coords.dim() != 2:
            raise ValueError(f"coords must be 2D [L,coord_dim], got {coords.shape}")
        self.coords = coords

    def forward(self, x, x_mark, subject_ids=None, mask=None):
        """
        返回形状与 DataEmbedding 一致：
        - 若 subject_embedding 存在：返回 [B, L+1, d_model]
        - 否则：返回 [B, L, d_model]
        """
        # 1) value embedding（与 DataEmbedding 一致）
        if self.joint_train:
            # per-subject embedding
            x = torch.stack([
                self.value_embedding[str(subject_id.item())](x[i])
                for i, subject_id in enumerate(subject_ids)
            ])
        else:
            x = self.value_embedding(x)  # [B,L,d_model]

        # 2) 加坐标 embedding（montage-aware 的唯一新增点）
        if self.use_coords:
            # coords: [L,coord_dim] -> [L,d_model] -> [B,L,d_model]
            if self.coords is None or self.coords.numel() == 0:
                # 没注入 coords 就当作 0，不让程序崩
                pass
            else:
                # 确保 coords 在同设备
                coords = self.coords.to(x.device)
                if coords.shape[0] != x.shape[1]:
                    raise ValueError(
                        f"coords length L mismatch: coords {coords.shape[0]} vs x tokens {x.shape[1]}"
                    )
                pos = self.coord_mlp(coords)  # [L,d_model]
                x = x + self.coord_scale * pos.unsqueeze(0)  # broadcast 到 batch

        # 3) temporal + positional（与 DataEmbedding 一致）
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark) + self.position_embedding(x)

        # 4) mask token（与 DataEmbedding 一致）
        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()

        # 5) subject token（与 DataEmbedding 一致）
        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(subject_ids)  # [B,1,d_model]
            x = torch.cat([subject_emb, x], dim=1)            # [B,L+1,d_model]

        return self.dropout(x)
