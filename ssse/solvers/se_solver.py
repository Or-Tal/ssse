from concurrent.futures import ThreadPoolExecutor
import os
import flashy
import omegaconf
from .base_solver import BaseSolver
from ..models.model_factory import model_factory
from ..data.data_factory import dataloaders_factory
import torch
from ..losses.loss_factory import loss_factory
from ..metrics.metrics_factory import get_pesq, get_snr, get_stoi
from ..optim.optimizer_factory import optimizer_factory


class SESolver(BaseSolver):

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.build_loss_function()

    def build_loss_function(self):
        self.loss_func = loss_factory(self.cfg)
    
    def loss_function(self, *args, **kwargs):
        return self.loss_func(*args, **kwargs)


    def build_model(self):
        """
        Method to implement to initialize model.
        """
        self.model = model_factory(self.cfg, self.cfg.model.model_class_name).to(self.device)
        

    def build_optimizers(self):
        self.optimizer = optimizer_factory(self.cfg, self.model)

    def build_dataloaders(self):
        """
        Method to implement to initialize dataloaders.
        """
        self.dataloaders = dataloaders_factory(self.cfg)
        self.train_updates_per_epoch = len(self.dataloaders['train'])
        if self.cfg.solver.optim.updates_per_epoch:
            self.train_updates_per_epoch = self.cfg.solver.optim.updates_per_epoch

    def optimize(self, loss):
        loss.backward()
        # if self.cfg.solver.optim.max_norm:
        #     torch.nn.utils.clip_grad_norm_(
        #         self.model.parameters(), self.cfg.solver.optim.max_norm
        #     )
        flashy.distrib.sync_model(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
        self.optimizer.step()
        if hasattr(self, "lr_scheduler") and self.lr_scheduler:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def _step(self, batch, metrics, is_training: bool):
        """
        Perform one training or valid step on a given batch.
        """
        if len(batch) > 2:
            noisy_sigs = batch[0]
            vad_mask = batch[-1]
        else:
            noisy_sigs, vad_mask = batch
        noisy_sigs = noisy_sigs.to(self.device)
        vad_mask = vad_mask.to(self.device)

        if not is_training:
            with torch.no_grad():
                outputs = self.model(noisy_sigs)
        else:
            outputs = self.model(noisy_sigs)
        # logger.info(f"outputs: {type(outputs)}, batch: {type(batch)}, device: {device}")

        losses = self.loss_function(outputs, noisy_sigs, vad_mask)
        loss = sum(losses)
        metrics['reconst'] = losses[0]
        metrics['cont'] = losses[1]
        metrics['reg'] = losses[2]
        if is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.optimize(loss)
        metrics['tot_loss'] = loss
        metrics['epoch'] = self.epoch

        return metrics

    def set_train(self):
        self.model.train()
    
    def set_eval(self):
        self.model.eval()

    def run_evaluation(self, batch, metrics):
        noisy_sigs, clean_sigs, vad_mask, _ = batch
        noisy_sigs = noisy_sigs.to(self.device)
        clean_sigs = clean_sigs.to(self.device)
        with torch.no_grad():
            estimate = self.model(noisy_sigs, eval=True)
        if clean_sigs.shape[-1] > estimate.shape[-1]:
            clean_sigs = clean_sigs[..., :estimate.shape[-1]]
            noisy_sigs = noisy_sigs[..., :estimate.shape[-1]]
        elif estimate.shape[-1] > clean_sigs.shape[-1]:
            estimate = estimate[..., :clean_sigs.shape[-1]]
        
        metrics['snr'] = get_snr(estimate, estimate - clean_sigs).item()

        estimate_numpy = estimate.cpu().squeeze(1).numpy()
        clean_numpy = clean_sigs.cpu().squeeze(1).numpy()

        metrics['pesq'] = get_pesq(clean_numpy, estimate_numpy, sr=self.cfg.dset.sample_rate)
        metrics['stoi'] = get_stoi(clean_numpy, estimate_numpy, sr=self.cfg.dset.sample_rate)

        return metrics


    def common_train_valid_evaluate(self, dataset_split: str):
        """
        Common logic for train and valid stages.
        """
        is_training = self.current_stage == 'train'
        do_eval = self.current_stage == 'evaluate'
        if is_training:
            self.set_train()
        else:
            self.set_eval()
        loader = self.dataloaders[dataset_split]
        # get a different order for distributed training, otherwise this will get ignored
        if flashy.distrib.world_size() > 1 \
           and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(self.epoch)

        updates_per_epoch = self.train_updates_per_epoch if is_training else len(loader)
        lp = self.log_progress(self.current_stage, loader, total=updates_per_epoch, updates=self.log_updates)
        average = flashy.averager()  # epoch wise average
        instant_average = flashy.averager()  # average between two logging

        metrics: dict = {}
        for idx, batch in enumerate(lp):
            if idx >= updates_per_epoch:
                break
            metrics = {}
            metrics = self.run_evaluation(batch, metrics) if do_eval else self._step(batch, metrics, is_training)
            instant_metrics = instant_average(metrics)
            if lp.update(**instant_metrics):
                instant_average = flashy.averager()  # reset averager between two logging
            metrics = average(metrics)  # epoch wise average
        metrics = flashy.distrib.average_metrics(metrics, updates_per_epoch)
        return metrics

    def show(self):
        """
        Method to log any information without running the job.
        """
        self.logger.info("Model:")
        return self.log_model_summary(self.model)

    def train(self):
        """
        Train stage.
        """
        return self.common_train_valid_evaluate('train')

    def valid(self):
        """
        Valid stage.
        """
        return self.common_train_valid_evaluate('valid')

    def evaluate(self):
        """
        Evaluate stage.
        """
        return self.common_train_valid_evaluate('evaluate')

    def generate(self):
        """
        Generate stage.
        """
        pass