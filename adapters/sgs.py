"""Global SGS over fixed reset rows; manipulation settings from arXiv:2610.12465.

No official implementation was released when this was written. Empty histories
have p=0 (hence all rows initially have equal weight). Only observed outcomes
enter the mean. Concurrent completions of one row are ordered by environment ID.
"""
import torch


class SuccessGuidedSampler:
    def __init__(self, size, device, history=100, temperature=2.0,
                 target=0.5, epsilon=1e-4, concentration=1.0):
        self.history = history
        self.temperature = max(temperature, 1.0)
        self.target = target
        self.epsilon = epsilon
        self.concentration = concentration
        self.outcomes = torch.zeros(size, history, dtype=torch.bool, device=device)
        self.total = torch.zeros(size, dtype=torch.long, device=device)
        self.successes = torch.zeros_like(self.total)
        self.weights = self._weights(torch.zeros(size, device=device))

    def _weights(self, p):
        e, k, t = self.epsilon, self.concentration, self.target
        w = (p + e).pow(k * t) * (1.0 - p + e).pow(k * (1.0 - t))
        # Equivalent to softmax(log(max(w, eps) + eps) / T), unnormalized.
        return (w.clamp_min(e) + e).pow(1.0 / self.temperature)

    @torch.no_grad()
    def update(self, rows, success):
        if rows.numel() == 0:
            return
        order = torch.argsort(rows, stable=True)
        rows, success = rows[order], success[order].bool()
        unique, counts = torch.unique_consecutive(rows, return_counts=True)
        starts = counts.cumsum(0) - counts
        rank = torch.arange(rows.numel(), device=rows.device) - torch.repeat_interleave(starts, counts)
        # A single batch may contain more than H results for a row. Retain only
        # its last H writes, avoiding duplicate-index writes and GPU races.
        keep = rank >= torch.repeat_interleave(counts, counts) - self.history
        kept_rows = rows[keep]
        slots = (self.total[kept_rows] + rank[keep]) % self.history
        delta = success[keep].long() - self.outcomes[kept_rows, slots].long()
        self.outcomes[kept_rows, slots] = success[keep]
        self.successes.scatter_add_(0, kept_rows, delta)
        self.total[unique] += counts
        p = self.successes[unique].float() / self.total[unique].clamp(max=self.history).float()
        self.weights[unique] = self._weights(p)

    def sample(self, count):
        return torch.multinomial(self.weights, count, replacement=True)

    def state_dict(self):
        return {name: getattr(self, name).clone() for name in ('outcomes', 'total', 'successes', 'weights')}

    def load_state_dict(self, state):
        for name, value in state.items():
            getattr(self, name).copy_(value)
