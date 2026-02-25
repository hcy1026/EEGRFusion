# rectified_flow_prior.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

class RFPipe:
    """
    Rectified Flow prior:
      - Train: flow matching on straight line between x0~N(0,I) and x1=h_embeds
      - Sample: Euler integrate dx/dt = v_theta(x,t,c)
    Keeps the same external interface as diffusion_prior.Pipe.
    """

    def __init__(self, diffusion_prior, device="cuda", t_scale=1000.0, cfg_drop_prob=0.1):
        self.diffusion_prior = diffusion_prior.to(device)
        self.device = device
        self.t_scale = float(t_scale)
        self.cfg_drop_prob = float(cfg_drop_prob)

    def train(self, dataloader, num_epochs=10, learning_rate=1e-4, warmup_steps=500):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction="none")
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        for epoch in range(num_epochs):
            loss_sum = 0.0
            for batch in dataloader:
                c_embeds = batch["c_embedding"].to(device) if "c_embedding" in batch else None
                x1 = batch["h_embedding"].to(device)  # target embedding (CLIP image feature)
                N = x1.shape[0]

                # CFG-style dropout for condition
                if c_embeds is not None and torch.rand(1, device=device) < self.cfg_drop_prob:
                    c_embeds = None

                # x0 ~ N(0, I)
                x0 = torch.randn_like(x1)

                # t ~ U(0,1)
                t = torch.rand(N, device=device, dtype=torch.float32)
                t_view = t.view(N, 1)

                # Straight path: x_t = (1-t)x0 + t x1
                x_t = (1.0 - t_view) * x0 + t_view * x1

                # Target velocity: v* = dx/dt = x1 - x0
                v_target = (x1 - x0)

                # Model input time embedding (keep compatible with your Timesteps module)
                t_in = t * self.t_scale  # shape [N]

                v_pred = self.diffusion_prior(x_t, t_in, c_embeds)

                loss = criterion(v_pred, v_target).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_sum += float(loss.item())

            print(f"[RF] epoch: {epoch}, loss: {loss_sum / max(1, len(dataloader)):.6f}")

    @torch.no_grad()
    def generate(
        self,
        c_embeds=None,
        num_inference_steps=16,
        timesteps=None,              # keep arg for compatibility; not used
        guidance_scale=5.0,
        generator=None,
    ):
        self.diffusion_prior.eval()
        device = self.device

        N = c_embeds.shape[0] if c_embeds is not None else 1
        if c_embeds is not None:
            c_embeds = c_embeds.to(device)

        # Start from x0 ~ N(0, I)
        x = torch.randn(N, self.diffusion_prior.embed_dim, device=device, generator=generator)

        # Euler integration from t=0 -> 1
        steps = int(num_inference_steps)
        dt = 1.0 / steps

        for i in range(steps):
            # Midpoint time works slightly better than left endpoint in practice
            t = (i + 0.5) / steps
            t_in = torch.full((N,), t * self.t_scale, device=device, dtype=torch.float32)

            if guidance_scale == 0 or c_embeds is None:
                v = self.diffusion_prior(x, t_in, None)
            else:
                v_cond = self.diffusion_prior(x, t_in, c_embeds)
                v_uncond = self.diffusion_prior(x, t_in, None)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)

            x = x + dt * v

        return x
