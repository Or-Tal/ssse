import torch
from .se_solver import SESolver

class SupSESolver(SESolver):

    def _step(self, batch, metrics, is_training: bool):
        """
        Perform one training or valid step on a given batch.
        """
        noisy_sigs, clean_sigs, vad_mask = batch
        noisy_sigs = noisy_sigs.to(self.device)
        vad_mask = vad_mask.to(self.device)

        if not is_training:
            with torch.no_grad():
                outputs = self.model(noisy_sigs)
        else:
            outputs = self.model(noisy_sigs)
        # logger.info(f"outputs: {type(outputs)}, batch: {type(batch)}, device: {device}")

        loss = self.loss_function(outputs, noisy_sigs, clean_sigs, vad_mask)
        if is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.optimize(loss)
        metrics['loss'] = loss

        return metrics