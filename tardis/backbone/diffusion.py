"""TabDiff: continuous-time mixed-type diffusion with learnable per-column noise schedules.

Adapted from the official implementation (https://github.com/MinkaiXu/TabDiff). Numerical
columns follow an EDM (Karras et al., 2022) process with a power-mean schedule per column;
categorical columns follow a masked (absorbing-state) process with a log-linear schedule
per column. The only addition is the ``guidance`` hook of :meth:`TabDiff.sample`, through
which Stage II perturbs the predicted score (Eq. 4).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S_CHURN = 1.0


class PowerMeanNoise(nn.Module):
    def __init__(self, num_numerical, sigma_min=0.002, sigma_max=80.0, rho_init=7.0, rho_offset=5.0):
        super().__init__()
        self.sigma_min, self.sigma_max, self.rho_offset = sigma_min, sigma_max, rho_offset
        self.rho_raw = nn.Parameter(torch.full((num_numerical,), rho_init))

    def rho(self):
        return F.softplus(self.rho_raw) + self.rho_offset

    def total_noise(self, t):
        rho = self.rho()
        lo, hi = self.sigma_min ** (1 / rho), self.sigma_max ** (1 / rho)
        return (lo + t * (hi - lo)).pow(rho)

    def inverse_to_t(self, sigma):
        rho = self.rho()
        lo, hi = self.sigma_min ** (1 / rho), self.sigma_max ** (1 / rho)
        return (sigma.pow(1 / rho) - lo) / (hi - lo)


class LogLinearNoise(nn.Module):
    def __init__(self, num_categories, eps_max=1e-3, eps_min=1e-5, k_init=-6.0, k_offset=1.0):
        super().__init__()
        self.eps_max, self.eps_min, self.k_offset = eps_max, eps_min, k_offset
        self.k_raw = nn.Parameter(torch.full((num_categories,), k_init))

    def k(self):
        return F.softplus(self.k_raw) + self.k_offset

    def rate_noise(self, t):
        k, c = self.k(), 1 - self.eps_max - self.eps_min
        return c * k * t.pow(k - 1) / (1 - (c * t.pow(k) + self.eps_min))

    def total_noise(self, t):
        return -torch.log1p(-((1 - self.eps_max - self.eps_min) * t.pow(self.k()) + self.eps_min))


class TabDiff(nn.Module):
    NEG_INF = -1e6

    def __init__(self, denoise_fn, d_num, categories, num_timesteps=50, sigma_data=1.0):
        super().__init__()
        self._denoise_fn = denoise_fn
        self.d_num = d_num
        self.num_classes = np.asarray(categories, dtype=int)
        self.num_timesteps = num_timesteps
        self.sigma_data = sigma_data
        self.num_schedule = PowerMeanNoise(d_num)
        self.cat_schedule = LogLinearNoise(len(categories))
        self.register_buffer("mask_index", torch.tensor(self.num_classes, dtype=torch.long))
        self.num_classes_w_mask = tuple(int(k) + 1 for k in self.num_classes)
        # the stochastic (churn) sampler and 2nd-order correction need numerical columns
        self.stochastic = self.second_order = d_num > 0

    @property
    def has_cat(self):
        return len(self.num_classes) > 0

    # ---------------------------------------------------------------- training
    def mixed_loss(self, x):
        x_num, x_cat = x[:, : self.d_num], x[:, self.d_num:].long()
        t = torch.rand(len(x), 1, device=x.device)
        sigma_num = self.num_schedule.total_noise(t)
        sigma_cat = self.cat_schedule.total_noise(t)
        dsigma_cat = self.cat_schedule.rate_noise(t)

        x_num_t = x_num + torch.randn_like(x_num) * sigma_num
        x_cat_t, x_cat_t_soft = x_cat, x_cat
        if self.has_cat:
            x_cat_t, x_cat_t_soft = self.q_xt_soft(x_cat, -torch.expm1(-sigma_cat))

        out_num, out_cat = self._denoise_fn(x_num_t, x_cat_t_soft, t.squeeze(-1), sigma=sigma_num)
        out_num, out_cat = out_num.float(), out_cat.float()

        c_loss = torch.zeros((), device=x.device)
        d_loss = torch.zeros((), device=x.device)
        if self.d_num > 0:
            w = (sigma_num ** 2 + self.sigma_data ** 2) / (sigma_num * self.sigma_data) ** 2
            c_loss = (w * (out_num - x_num) ** 2).mean()
        if self.has_cat:
            logits = self._subs_parameterization(out_cat, x_cat_t)
            log_p = torch.gather(logits, -1, x_cat[:, :, None]).squeeze(-1)
            d_loss = (-dsigma_cat / torch.expm1(sigma_cat) * log_p).mean()
        return d_loss, c_loss

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, guidance=None):
        """Draw ``n`` rows. ``guidance`` (optional) perturbs the denoiser output at the
        steps where ``guidance.active(i)``; see :class:`tardis.guidance.ChamferGuidance`."""
        device = self.mask_index.device
        t = torch.linspace(0, 1, self.num_timesteps, device=device)[:, None]
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[:-1]

        if self.stochastic:
            gamma = min(S_CHURN / self.num_timesteps, np.sqrt(2) - 1)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat).min(dim=-1, keepdim=True).values
            out = (t_hat > 1).squeeze(-1)
            sigma_num_hat[out], t_hat[out] = sigma_num_cur[out], t[out]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat, sigma_num_hat, sigma_cat_hat = t, sigma_num_cur, sigma_cat_cur

        z_num = torch.randn(n, self.d_num, device=device) * sigma_num_cur[-1]
        z_cat = self.mask_index[None].repeat(n, 1) if self.has_cat else torch.zeros(n, 0, device=device)
        if guidance is not None:
            guidance.reset()
        for i in reversed(range(self.num_timesteps)):
            z_num, z_cat = self._edm_update(
                z_num, z_cat, i, t[i - 1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i],
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                guidance if guidance is not None and guidance.active(i) else None,
            )
        return torch.cat([z_num, z_cat.float()], dim=1)

    def sample_all(self, n, batch_size=10000, guidance=None):
        """Sample ``n`` rows in batches, discarding rows with NaNs."""
        chunks, done = [], 0
        while done < n:
            x = self.sample(min(batch_size, n - done), guidance).cpu()
            x = x[~torch.isnan(x).any(dim=1)]
            chunks.append(x)
            done += len(x)
        return torch.cat(chunks)[:n]

    def _edm_update(self, x_num, x_cat, i, t_next, t_hat, s_num, s_num_next, s_num_hat, s_cat, s_cat_next, s_cat_hat, guide):
        b = len(x_num)
        x_num_hat = x_num + (s_num_hat ** 2 - s_num ** 2).sqrt() * torch.randn_like(x_num)
        x_cat_hat = x_cat
        if self.has_cat:
            x_cat_hat = self.q_xt(x_cat, -torch.expm1(s_cat - s_cat_hat))

        denoised, raw_logits = self._denoise(x_num_hat, x_cat_hat, t_hat.repeat(b), s_num_hat.repeat(b, 1), i, guide)
        d_cur = (x_num_hat - denoised) / s_num_hat
        x_num_next = x_num_hat + (s_num_next - s_num_hat) * d_cur

        x_cat_next = x_cat
        if self.has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-s_cat_hat).repeat(b, 1)
            alpha_s = torch.exp(-s_cat_next).repeat(b, 1)
            x_cat_next = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)

        if self.second_order and i > 0:
            denoised, _ = self._denoise(x_num_next, x_cat_hat, t_next.repeat(b), s_num_next.repeat(b, 1), i, guide)
            d_prime = (x_num_next - denoised) / s_num_next
            x_num_next = x_num_hat + (s_num_next - s_num_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_num_next, x_cat_next

    def _denoise(self, x_num, x_cat, t, sigma, i, guide):
        x_cat_oh = self.to_one_hot(x_cat).float() if self.has_cat else x_cat
        if guide is None or self.d_num == 0:
            return self._denoise_fn(x_num.float(), x_cat_oh, t, sigma=sigma)
        # Stage II (Eq. 4): perturb the prediction with the gradient of the Chamfer functional,
        # evaluated on the predicted clean sample and taken w.r.t. the noisy numerical input.
        with torch.enable_grad():
            x = x_num.detach().float().requires_grad_(True)
            denoised, raw_logits = self._denoise_fn(x, x_cat_oh, t, sigma=sigma)
            x0_cat = x_cat
            if self.has_cat:  # predicted clean categories: kept where unmasked, argmax where masked
                x0_cat = self._subs_parameterization(raw_logits.detach(), x_cat).argmax(-1)
            grad = guide.gradient(x, denoised, x0_cat)
        return denoised.detach() - guide.gamma(i) * grad, raw_logits.detach()

    # ------------------------------------------------- masked categorical diffusion
    def to_one_hot(self, x_cat):
        return torch.cat([F.one_hot(x_cat[:, j].long(), k) for j, k in enumerate(self.num_classes_w_mask)], dim=-1)

    def q_xt(self, x, move_chance):
        return torch.where(torch.rand_like(x, dtype=torch.float) < move_chance, self.mask_index, x)

    def q_xt_soft(self, x, move_chance):
        """Forward masking with straight-through Gumbel samples (learnable schedule)."""
        xt_soft = torch.zeros(len(x), sum(self.num_classes_w_mask), device=x.device)
        xt = torch.zeros_like(x)
        offset = 0
        for j, k in enumerate(self.num_classes_w_mask):
            logp = torch.log(torch.stack([1 - move_chance[:, j], move_chance[:, j]], dim=-1))
            s = F.gumbel_softmax(logp, tau=0.01, hard=True)
            idx = torch.stack([x[:, j] + offset, torch.full_like(x[:, j], offset + k - 1)], dim=-1)
            xt_soft[torch.arange(len(x)).unsqueeze(1), idx] = s
            xt[:, j] = torch.where(s[:, 1] > s[:, 0], self.mask_index[j], x[:, j])
            offset += k
        return xt, xt_soft

    def pad(self, x, value):
        max_k = max(self.num_classes_w_mask)
        parts = torch.split(x, self.num_classes_w_mask, dim=-1)
        return torch.stack([F.pad(p, (0, max_k - p.shape[-1]), value=value) for p in parts], dim=-2)

    def _subs_parameterization(self, raw, xt):
        """Log-probabilities of the clean category; [MASK] gets -inf and unmasked entries are kept."""
        logits = self.pad(raw, self.NEG_INF)
        logits[:, range(logits.shape[1]), self.mask_index] += self.NEG_INF
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked = xt != self.mask_index
        logits[unmasked] = self.NEG_INF
        logits[unmasked, xt[unmasked]] = 0
        return logits

    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        move_t, move_s = (1 - alpha_t).unsqueeze(-1), (1 - alpha_s).unsqueeze(-1)
        q_xs = log_p_x0.exp() * (move_t - move_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_s[:, :, 0]
        valid = torch.arange(q_xs.shape[-1], device=x.device)[None] <= self.mask_index[:, None]
        q_xs = q_xs * valid
        gumbel = 1e-10 - (torch.rand_like(q_xs) + 1e-10).log()
        sampled = (q_xs / gumbel).argmax(dim=-1)
        return torch.where(x != self.mask_index, x, sampled)
