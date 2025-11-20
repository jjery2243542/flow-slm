import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FlowLoss(nn.Module):
    def __init__(self, target_dim, z_dim, net, sigma_min=1e-5, t_dist="uniform", null_prob=0.05, x_pred=False, min_clip=0.05):
        super().__init__()
        self.target_dim = target_dim
        self.z_dim = z_dim
        self.sigma_min = sigma_min
        self.min_clip = min_clip
        self.t_dist = t_dist
        self.null_prob = null_prob
        self.null_emb = torch.nn.Embedding(1, z_dim)
        self.x_pred = x_pred

        # output layer for loss computation
        self.net = net 

    def forward(self, z, target):
        if self.t_dist == "uniform":
            t = torch.rand([target.shape[0], target.shape[1]], device=target.device, dtype=target.dtype)
        elif self.t_dist == "logit_normal": 
            t = torch.sigmoid(torch.randn([target.shape[0], target.shape[1]], device=target.device, dtype=target.dtype))
        else:
            raise NotImplementedError(f"t_dist {self.t_dist} not implemented")

        noise = torch.randn_like(target)
        t_expand = t.unsqueeze(2)
        if self.x_pred:
            psi_t = t_expand * target + (1 - t_expand) * noise
            # hard-coded from JIT paper 
            u = (target - noise) / torch.clip(1 - t_expand, min=self.min_clip)
        else:
            psi_t = (1 - (1 - self.sigma_min) * t_expand) * noise + t_expand * target
            u = target - (1 - self.sigma_min) * noise

        # use CFG during training
        if self.training:
            sample_null = torch.rand(z.shape[:2], device=target.device, dtype=target.dtype)
            is_null = (sample_null < self.null_prob).to(target.device, dtype=target.dtype)
            z = z * (1 - is_null).unsqueeze(2) + self.null_emb.weight * is_null.unsqueeze(2)

        out = self.net(psi_t, t, z)

        if self.x_pred:
            v_pred = (out - psi_t) / torch.clip(1 - t_expand, min=self.min_clip)
            loss = F.mse_loss(v_pred, u, reduction='none')
        else:
            loss = F.mse_loss(out, u, reduction='none')
        return loss

    def sample(self, z, x=None, steps=100, temperature=1.0, schedule="linear", truncation=1.0, solver="euler", cfg_scale=0.0, shift_alpha=2.0):
        if x is None:
            x = torch.randn(z.shape[0], z.shape[1], self.target_dim, device=z.device, dtype=z.dtype)

            if truncation < 1.0:
                while torch.any((x > truncation) | (x < -truncation)): 
                    x[x.abs() > truncation] = torch.randn_like(x[x.abs() > truncation])
            x = x * temperature

        if schedule == "linear":
            t_span = torch.linspace(0, 1, steps + 1, device=z.device, dtype=z.dtype)
        elif schedule == "shifted_linear":
            t_span = torch.linspace(0, 1, steps + 1, device=z.device, dtype=z.dtype)
            t_span = (t_span * shift_alpha) / (1 + (shift_alpha - 1) * t_span)
        else:
            raise NotImplementedError(f"schedule {schedule} not implemented")

        if solver == "euler":
            t, dt = t_span[0], t_span[1] - t_span[0]
            t_expand = t.expand(x.shape[0], x.shape[1])

            sols = []
            for step in range(1, len(t_span)):
                dphi_dt = self._compute_dphi_dt(x, t_expand, z, cfg_scale)
                x = x + dt * dphi_dt
                t = t + dt
                t_expand = t.expand(x.shape[0], x.shape[1])
                sols.append(x)
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t

            return sols[-1]

        elif solver == "midpoint":
            from torchdiffeq import odeint

            def ode_func(t, x):
                t_expand = t.expand(x.shape[0], x.shape[1])
                return self._compute_dphi_dt(x, t_expand, z, cfg_scale)

            sols = odeint(ode_func, x, t_span, method='midpoint')
            return sols[-1]

        elif solver == "adaptive_heun":
            from torchdiffeq import odeint

            def ode_func(t, x):
                t_expand = t.expand(x.shape[0], x.shape[1])
                return self._compute_dphi_dt(x, t_expand, z, cfg_scale)

            t_span = x.new_tensor([0.0, 1.0])
            sols = odeint(ode_func, x, t_span, method='adaptive_heun', atol=1e-4, rtol=1e-4)
            return sols[-1]

        elif solver == "dopri5":
            from torchdiffeq import odeint
            #fevals = 0
            def ode_func(t, x):
                #nonlocal fevals 
                t_expand = t.expand(x.shape[0], x.shape[1])
                #fevals += 1
                return self._compute_dphi_dt(x, t_expand, z, cfg_scale)

            t_span = x.new_tensor([0.0, 1.0])
            sols = odeint(ode_func, x, t_span, method='dopri5', atol=1e-4, rtol=1e-4)
            return sols[-1]

    def _compute_dphi_dt(self, x, t_expand, z, cfg_scale=0.0):
        """
        Compute dphi_dt with optional classifier-free guidance (batch concatenation).
        This project doesn't require the 'feature' mode, so all CFG is done by
        doubling the batch dimension (cond + uncond) and then combining outputs.
        """
        if cfg_scale <= 0.0 and not self.x_pred:
            return self.net(x, t_expand, z)
        elif cfg_scale <= 0.0 and self.x_pred:
            x_pred = self.net(x, t_expand, z)
            return (x_pred - x) / torch.clip(1 - t_expand.unsqueeze(2))

        # prepare null embedding expanded to z shape and on correct device/dtype
        null_z = self.null_emb.weight.expand(z.shape[0], z.shape[1], -1).to(z.device, dtype=z.dtype)

        # batch-concatenate conditional and unconditional examples
        z_concat = torch.cat([z, null_z], dim=0)
        x_concat = torch.cat([x, x], dim=0)
        t_concat = torch.cat([t_expand, t_expand], dim=0)
        if not self.x_pred:
            dphi_dt = self.net(x_concat, t_concat, z_concat)
            dphi_dt_cond, dphi_dt_uncond = torch.chunk(dphi_dt, 2, dim=0)
        else:
            x_pred = self.net(x_concat, t_concat, z_concat)
            dphi_dt = (x_pred - x_concat) / torch.clip(1 - t_concat.unsqueeze(2), min=self.min_clip)
            dphi_dt_cond, dphi_dt_uncond = torch.chunk(dphi_dt, 2, dim=0)
        return dphi_dt_cond + cfg_scale * (dphi_dt_cond - dphi_dt_uncond)
