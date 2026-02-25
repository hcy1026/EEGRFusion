import torch
import torch.nn as nn

# 复用你现有 Embed.py 里的实现，确保行为一致
from .Embed import PositionalEmbedding, TemporalEmbedding, SubjectEmbedding, TimeFeatureEmbedding


class MontageAwareEmbedding(nn.Module):
    """
    Montage-aware 的 DataEmbedding（以“每个电极/通道为一个 token”为前提）。

    增强点：
      1) 坐标 Fourier features + MLP
      2) 基于几何距离的 kNN 图消息传递（空间邻域混合）
      3) FiLM（用坐标调制 value embedding 的缩放/平移）
      4) 粗粒度区域离散嵌入（左/中/右；前/中/后；上/中/下）

    接口兼容原始 DataEmbedding：
        forward(x, x_mark, subject_ids=None, mask=None) -> [B, L(+1), d_model]
    并提供 set_coords(coords) 用于注入电极坐标。
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

        # montage-aware
        coord_dim: int = 3,
        coord_scale: float = 1.0,
        use_coords: bool = True,

        # stronger coord encoding
        use_fourier: bool = True,
        fourier_freqs: int = 6,
        fourier_base: float = 2.0,

        # graph mixing
        use_graph: bool = True,
        graph_topk: int = 8,
        graph_sigma: float = 0.4,  # 最初0.6 0.4
        use_edge_bias: bool = True,
        graph_scale: float = 0.8,  # 最初0.5 0.8

        # FiLM modulation
        use_film: bool = True,
        film_scale: float = 1.0,

        # region embedding
        use_region_embed: bool = True,
        region_threshold: float = 0.5,
        region_scale: float = 0.2,
    ):
        super().__init__()

        self.joint_train = joint_train
        self.use_coords = use_coords
        self.coord_dim = coord_dim

        # value embedding（与 DataEmbedding 保持一致）
        if joint_train and num_subjects is not None:
            self.value_embedding = nn.ModuleDict({
                str(subject_id): nn.Linear(c_in, d_model) for subject_id in range(num_subjects)
            })
        else:
            self.value_embedding = nn.Linear(c_in, d_model)

        # pos / temporal / dropout / subject / mask（与 DataEmbedding 对齐）
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        self.mask_token = nn.Parameter(torch.randn(1, d_model))

        # ---------------- montage-aware components ----------------
        self.use_fourier = use_fourier
        self.fourier_freqs = int(fourier_freqs)
        self.fourier_base = float(fourier_base)

        coord_feat_dim = coord_dim
        if use_fourier:
            coord_feat_dim = coord_dim * (1 + 2 * self.fourier_freqs)

        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # coord 注入强度（可学习）
        self.coord_scale = nn.Parameter(torch.tensor(float(coord_scale)))

        # FiLM
        self.use_film = use_film and use_coords
        if self.use_film:
            self.film = nn.Linear(d_model, 2 * d_model)
            self.film_scale = nn.Parameter(torch.tensor(float(film_scale)))

        # Graph mixing（kNN）
        self.use_graph = use_graph and use_coords
        self.graph_topk = int(graph_topk)
        self.use_edge_bias = use_edge_bias and use_coords
        if self.use_graph:
            self.graph_sigma = nn.Parameter(torch.tensor(float(graph_sigma)))
            self.graph_scale = nn.Parameter(torch.tensor(float(graph_scale)))
            self.graph_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.graph_gate = nn.Linear(2 * d_model, d_model)

            if self.use_edge_bias:
                self.edge_mlp = nn.Sequential(
                    nn.Linear(coord_dim, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )

        # Region embedding
        self.use_region_embed = use_region_embed and use_coords
        self.region_threshold = float(region_threshold)
        if self.use_region_embed:
            self.region_scale = nn.Parameter(torch.tensor(float(region_scale)))
            self.hemi_emb = nn.Embedding(3, d_model)  # x: left/mid/right
            self.ap_emb = nn.Embedding(3, d_model)    # y: posterior/mid/anterior
            self.si_emb = nn.Embedding(3, d_model)    # z: inferior/mid/superior

        # buffers：set_coords 注入
        self.register_buffer("coords", torch.zeros(1, coord_dim), persistent=False)
        self.register_buffer("coords_norm", torch.zeros(1, coord_dim), persistent=False)
        self.register_buffer("perm", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("inv_perm", torch.empty(0, dtype=torch.long), persistent=False)

        # graph buffers
        self.register_buffer("knn_idx", torch.zeros(1, 1, dtype=torch.long), persistent=False)
        self.register_buffer("knn_d2", torch.zeros(1, 1), persistent=False)
        self.register_buffer("knn_rel", torch.zeros(1, 1, coord_dim), persistent=False)

        # region buffers
        self.register_buffer("hemi_id", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("ap_id", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("si_id", torch.zeros(1, dtype=torch.long), persistent=False)

    def _fourier_encode(self, coords: torch.Tensor) -> torch.Tensor:
        feats = [coords]
        if self.fourier_freqs <= 0:
            return coords
        freqs = (self.fourier_base ** torch.arange(self.fourier_freqs, device=coords.device, dtype=coords.dtype))
        freqs = freqs.view(1, 1, -1)
        x = coords.unsqueeze(-1)
        angles = 2.0 * torch.pi * x * freqs
        feats.append(torch.sin(angles).flatten(1))
        feats.append(torch.cos(angles).flatten(1))
        return torch.cat(feats, dim=1)

    @torch.no_grad()
    def set_coords(self, coords: torch.Tensor):
        if coords.dim() != 2:
            raise ValueError(f"coords must be 2D [L,coord_dim], got {coords.shape}")
        if coords.size(1) != self.coord_dim:
            raise ValueError(f"coord_dim mismatch: expected {self.coord_dim}, got {coords.size(1)}")

        self.coords = coords

        # z-score 归一化，减弱单位尺度/平移影响
        c = coords.float()
        c = c - c.mean(dim=0, keepdim=True)
        c = c / (c.std(dim=0, keepdim=True) + 1e-6)
        self.coords_norm = c

        L = c.size(0)

        # ---- spatial permutation: make sequence order follow spatial adjacency ----
        d = torch.cdist(c, c, p=2)  # [L,L]
        start = d.sum(dim=1).argmin().item()  # 最“中心”的点作为起点
        visited = torch.zeros(L, dtype=torch.bool, device=c.device)
        perm = torch.empty(L, dtype=torch.long, device=c.device)

        cur = start
        for t in range(L):
            perm[t] = cur
            visited[cur] = True
            if t == L - 1:
                break
            dd = d[cur].clone()
            dd[visited] = 1e9
            cur = dd.argmin().item()

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(L, device=c.device)

        self.perm = perm
        self.inv_perm = inv_perm

        # 把 coords_norm / 后续图结构都建立在 perm 后的顺序上
        c = c[perm]
        self.coords_norm = c

        # kNN graph
        if self.use_graph:
            d = torch.cdist(c, c, p=2)
            d.fill_diagonal_(1e9)
            k = min(self.graph_topk, L - 1)
            knn = torch.topk(d, k=k, dim=1, largest=False)
            idx = knn.indices
            d2 = (knn.values ** 2)
            rel = c[idx] - c.unsqueeze(1)

            self.knn_idx = idx
            self.knn_d2 = d2
            self.knn_rel = rel

        # region ids
        if self.use_region_embed:
            thr = self.region_threshold

            def _to_3cls(v):
                return torch.where(v > thr, torch.full_like(v, 2, dtype=torch.long),
                                   torch.where(v < -thr, torch.full_like(v, 0, dtype=torch.long),
                                               torch.full_like(v, 1, dtype=torch.long)))

            self.hemi_id = _to_3cls(c[:, 0])
            self.ap_id = _to_3cls(c[:, 1])
            self.si_id = _to_3cls(c[:, 2])
        print("[MontageAware] coords_norm mean/std:",
              self.coords_norm.mean(dim=0), self.coords_norm.std(dim=0))

    def graph_refine(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D] (不含 subject token 的 63 tokens)
        复用 set_coords() 预计算的 knn_idx/knn_d2/knn_rel 做一次图混合
        """
        if not (self.use_graph and self.knn_idx.numel() > 0):
            return x

        idx = self.knn_idx.to(x.device)
        d2 = self.knn_d2.to(x.device, dtype=x.dtype)
        rel = self.knn_rel.to(x.device, dtype=x.dtype)

        sigma = torch.clamp(self.graph_sigma, min=1e-3)
        w = torch.exp(-d2 / (2.0 * sigma * sigma))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

        x_nb = x[:, idx, :]  # [B,L,k,D]
        if self.use_edge_bias:
            e = self.edge_mlp(rel)  # [L,k,D]
            x_nb = x_nb + e.unsqueeze(0)

        nb = (x_nb * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)  # [B,L,D]
        nb = self.graph_proj(nb)

        gate = torch.sigmoid(self.graph_gate(torch.cat([x, nb], dim=-1)))
        x = x + self.graph_scale * gate * nb
        return x

    def forward(self, x, x_mark, subject_ids=None, mask=None):
        # 1) value embedding
        if self.joint_train:
            if subject_ids is None:
                raise ValueError("joint_train=True requires subject_ids")
            x = torch.stack([
                self.value_embedding[str(subject_id.item())](x[i])
                for i, subject_id in enumerate(subject_ids)
            ])
        else:
            x = self.value_embedding(x)

        B, L, D = x.shape

        # 2) montage-aware
        if self.use_coords and self.coords_norm is not None and self.coords_norm.numel() > 0:
            coords = self.coords_norm.to(x.device, dtype=x.dtype)
            if coords.shape[0] != L:
                raise ValueError(f"coords length mismatch: coords {coords.shape[0]} vs tokens {L}")

            coord_in = self._fourier_encode(coords) if self.use_fourier else coords
            coord_emb = self.coord_mlp(coord_in)  # [L,D]

            # (a) additive injection
            x = x + self.coord_scale * coord_emb.unsqueeze(0)

            # (b) region prior
            if self.use_region_embed and self.hemi_id.numel() == L:
                hemi = self.hemi_emb(self.hemi_id.to(x.device))
                ap = self.ap_emb(self.ap_id.to(x.device))
                si = self.si_emb(self.si_id.to(x.device))
                x = x + self.region_scale * (hemi + ap + si).unsqueeze(0)

            # (c) FiLM
            if self.use_film:
                gb = self.film(coord_emb)
                gamma, beta = gb.chunk(2, dim=-1)
                s = self.film_scale
                x = x * (1.0 + s * torch.tanh(gamma).unsqueeze(0)) + s * beta.unsqueeze(0)

            # (d) kNN graph mixing
            if self.use_graph and self.knn_idx.numel() > 0:
                idx = self.knn_idx.to(x.device)
                d2 = self.knn_d2.to(x.device, dtype=x.dtype)
                rel = self.knn_rel.to(x.device, dtype=x.dtype)

                sigma = torch.clamp(self.graph_sigma, min=1e-3)
                w = torch.exp(-d2 / (2.0 * sigma * sigma))
                w = w / (w.sum(dim=1, keepdim=True) + 1e-6)


                x_nb = x[:, idx, :]  # [B,L,k,D]
                if self.use_edge_bias:
                    e = self.edge_mlp(rel)  # [L,k,D]
                    x_nb = x_nb + e.unsqueeze(0)

                nb = (x_nb * w.unsqueeze(0).unsqueeze(-1)).sum(dim=2)  # [B,L,D]
                nb = self.graph_proj(nb)

                gate = torch.sigmoid(self.graph_gate(torch.cat([x, nb], dim=-1)))
                x = x + self.graph_scale * gate * nb

                # # ---- DEBUG: print entropy & gate stats (throttled) ----
                # if self.training:
                #     if not hasattr(self, "_dbg_step"):
                #         self._dbg_step = 0
                #     if self._dbg_step % 200 == 0:
                #         entropy = -(w * (w + 1e-9).log()).sum(dim=1).mean()
                #         # 先算 gate（下面本来就要算），顺便看 gate 是否把图分支“关掉”
                #         # 注意：这里先不改你原逻辑，只做观测
                #         # gate 需要用到 nb，所以放在算完 nb 之后也可以
                #         print(
                #             f"[MontageAware] knn_entropy={entropy.item():.4f} "
                #             f"sigma={float(sigma.detach().cpu()):.4f} "
                #             f"graph_scale={float(self.graph_scale.detach().cpu()):.4f} "
                #             f"coord_scale={float(self.coord_scale.detach().cpu()):.4f}"
                #             f"gate_mean={float(gate.mean().detach().cpu()):.4f}"
                #         )
                #     self._dbg_step += 1

        # 3) position embedding 永远加；temporal 只有 x_mark 有才加
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark) + self.position_embedding(x_mark)

        # 4) mask token
        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()

        # 5) subject token
        if self.subject_embedding is not None:
            if subject_ids is None:
                raise ValueError("num_subjects is set but subject_ids is None")
            subject_emb = self.subject_embedding(subject_ids)
            x = torch.cat([subject_emb, x], dim=1)

        return self.dropout(x)
