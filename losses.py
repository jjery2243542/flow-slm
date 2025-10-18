import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FlowLoss(nn.Module):
    def __init__(self, target_dim, z_dim, net, sigma_min=1e-5, t_dist="uniform", null_prob=0.05):
        super().__init__()
        self.target_dim = target_dim
        self.z_dim = z_dim
        self.sigma_min = sigma_min
        self.t_dist = t_dist
        self.null_prob = null_prob
        self.null_emb = torch.nn.Embedding(1, z_dim)

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
        psi_t = (1 - (1 - self.sigma_min) * t_expand) * noise + t_expand * target
        u = target - (1 - self.sigma_min) * noise

        # use CFG during training
        if self.training:
            sample_null = torch.rand(z.shape[:2], device=target.device, dtype=target.dtype)
            is_null = (sample_null < self.null_prob).to(target.device, dtype=target.dtype)
            z = z * (1 - is_null).unsqueeze(2) + self.null_emb.weight * is_null.unsqueeze(2)

        out = self.net(psi_t, t, z)
        loss = F.mse_loss(out, u, reduction='none')
        return loss

    def sample(self, z, x=None, steps=100, temperature=1.0, schedule="linear", truncation=1.0, solver="euler", cfg_scale=0.0):
        if x is None:
            x = torch.randn(z.shape[0], z.shape[1], self.target_dim, device=z.device, dtype=z.dtype)

            if truncation < 1.0:
                while torch.any((x > truncation) | (x < -truncation)): 
                    x[x.abs() > truncation] = torch.randn_like(x[x.abs() > truncation])
            x = x * temperature

        if schedule == "linear":
            t_span = torch.linspace(0, 1, steps + 1, device=z.device, dtype=z.dtype)
        else:
            raise NotImplementedError(f"schedule {schedule} not implemented")

        if solver == "euler":
            t, dt = t_span[0], t_span[1] - t_span[0]
            t_expand = t.expand(x.shape[0], x.shape[1])

            sols = []
            for step in range(1, len(t_span)):
                if cfg_scale > 0.0:
                    z_concat = torch.cat([z, self.null_emb.weight.expand(z.shape[0], z.shape[1], -1).to(z.device, dtype=z.dtype)], dim=0)
                    x_concat = torch.cat([x, x], dim=0)
                    t_concat = torch.cat([t_expand, t_expand], dim=0)
                    dphi_dt = self.net(x_concat, t_concat, z_concat)
                    dphi_dt, dphi_dt_uncond = torch.chunk(dphi_dt, 2, dim=0)
                    dphi_dt = dphi_dt + cfg_scale * (dphi_dt - dphi_dt_uncond)
                else:
                    dphi_dt = self.net(x, t_expand, z)
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
                if cfg_scale > 0.0:
                    z_concat = torch.cat([z, self.null_emb.weight.expand(z.shape[0], z.shape[1], -1).to(z.device, dtype=z.dtype)], dim=0)
                    x_concat = torch.cat([x, x], dim=0)
                    t_concat = torch.cat([t_expand, t_expand], dim=0)
                    dphi_dt = self.net(x_concat, t_concat, z_concat)
                    dphi_dt, dphi_dt_uncond = torch.chunk(dphi_dt, 2, dim=0)
                    dphi_dt = dphi_dt + cfg_scale * (dphi_dt - dphi_dt_uncond)
                else:
                    dphi_dt = self.net(x, t_expand, z)
                return dphi_dt

            sols = odeint(ode_func, x, t_span, method='midpoint')
            return sols[-1]

        elif solver == "adaptive_heun":
            from torchdiffeq import odeint

            def ode_func(t, x):
                t_expand = t.expand(x.shape[0], x.shape[1])
                if cfg_scale > 0.0:
                    z_concat = torch.cat([z, self.null_emb.weight.expand(z.shape[0], z.shape[1], -1).to(z.device, dtype=z.dtype)], dim=0)
                    x_concat = torch.cat([x, x], dim=0)
                    t_concat = torch.cat([t_expand, t_expand], dim=0)
                    dphi_dt = self.net(x_concat, t_concat, z_concat)
                    dphi_dt, dphi_dt_uncond = torch.chunk(dphi_dt, 2, dim=0)
                    dphi_dt = dphi_dt + cfg_scale * (dphi_dt - dphi_dt_uncond)
                else:
                    dphi_dt = self.net(x, t_expand, z)
                return dphi_dt

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
                if cfg_scale > 0.0:
                    z_concat = torch.cat([z, self.null_emb.weight.expand(z.shape[0], z.shape[1], -1).to(z.device, dtype=z.dtype)], dim=2)
                    dphi_dt = self.net(x, t_expand, z_concat)
                    dphi_dt = dphi_dt[:, :, :self.z_dim]
                    dphi_dt_uncond = dphi_dt[:, :, self.z_dim:]
                    dphi_dt = dphi_dt + cfg_scale * (dphi_dt - dphi_dt_uncond)
                else:
                    dphi_dt = self.net(x, t_expand, z)
                return dphi_dt

            t_span = x.new_tensor([0.0, 1.0])
            sols = odeint(ode_func, x, t_span, method='dopri5', atol=1e-4, rtol=1e-4)
            return sols[-1]
