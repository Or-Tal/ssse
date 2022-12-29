"""Similar to the Balancer idea used for balancing out adversarial and objective
reconstruction metrics, this does the same for residual skip connection, ensuring
that a large part of the gradient comes from the skip, and is not overflown by
the residual block.
"""
from functools import partial
import typing as tp
import torch

import flashy


class ResidualBalancer(torch.nn.Module):
    """
    Balancer for residual skip connections. Should be used as follow.

        class BalancedResBlock(nn.Module):
            def __init__(self):
                ...
                self.balancer = ResidualBalancer()

            def forward(self, x):
                x_for_skip, x_for_res_block = self.balancer(x)
                return x_for_skip + self.block(x_for_res_block)

    This balancer will ensure that during the backward, the gradient
    coming from `x_for_res_block` doesn't represent more than X% of the
    overall gradient flowing back to `x`.

    ..Warning:: This is an experimental feature, and subject to API change.

    Args:
        max_residual_ratio (float): Maximum fraction of the gradient that
            can come from the residual skip. Gradient flow coming from the
            residual block will be upscaled if necessary. Passing 1. will
            result in a no-op.
        ema_decay (float): Exponential moving average decay factor for computing
            the average norms of the gradients coming from each gradient path.
    """

    def __init__(self, max_residual_ratio: float = 0.15, ema_decay: float = 0.9):
        super().__init__()
        assert max_residual_ratio <= 1
        assert max_residual_ratio >= 0
        self.max_residual_ratio = max_residual_ratio
        self.gradient_norms_averager = flashy.averager(ema_decay)
        self._pending_gradients: tp.Dict[str, torch.Tensor] = {}
        self._pending_metrics: tp.Dict[str, float] = {}

    def collect_extra_metrics(self) -> tp.Dict[str, float]:
        metrics = self._pending_metrics
        self._pending_metrics = {}
        return metrics

    def _log_backward_hook(self, grad: torch.Tensor, name: str) -> None:
        assert name not in self._pending_gradients
        self._pending_gradients[name] = grad

    def _rescale_backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        grads = dict(self._pending_gradients)
        self._pending_gradients.clear()
        assert 'skip' in grads
        assert 'res_block' in grads

        norms = {key: grad.data.norm(p=2) for key, grad in grads.items()}
        norms = flashy.distrib.average_metrics(norms)
        avg_norms = self.gradient_norms_averager(norms)
        sum_avg_norms = sum(avg_norms.values())

        factor: float = 1.
        if avg_norms['res_block'] > self.max_residual_ratio * sum_avg_norms:
            # We need to downscale the res block gradient.
            # Let us note g_s the gradient of the skip and g_rb the gradient
            # of the residual block. We further note <|g_*|> the moving average
            # tracking their norms. We want to find `factor` such that
            # factor * <|g_rb|> / (<|g_s|> + factor * <|g_rb|>) <= self.max_residual_ratio,
            factor = avg_norms['skip'] * self.max_residual_ratio / (
                avg_norms['res_block'] * (1 - self.max_residual_ratio))
            assert factor < 1.
        grad = grads['skip'] + factor * grads['res_block']
        self._pending_metrics['avg_norm_skip'] = avg_norms['skip']
        self._pending_metrics['avg_norm_res_block'] = avg_norms['res_block']
        self._pending_metrics['factor'] = factor
        return grad

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Given the input `x`, this returns 2 variables `x_for_skip`, and
        `x_for_res_block`. They both have the same value but will
        handle the backward differently, recording individual gradients coming
        through.
        """
        if self.max_residual_ratio == 1 or not x.requires_grad:
            # Noop
            return x, x

        x_for_skip = x.clone()
        x_for_skip.register_hook(partial(self._log_backward_hook, name='skip'))
        x_for_res_block = x.clone()
        x_for_res_block.register_hook(partial(self._log_backward_hook, name='res_block'))
        x.register_hook(self._rescale_backward_hook)
        return x_for_skip, x_for_res_block
