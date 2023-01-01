# this base class was loaned from 
# https://github.com/fairinternal/audiocraft/blob/main/magma/magma/solvers/base.py

from abc import ABC, abstractmethod
import typing as tp
import flashy
import omegaconf
import torch
from torch import nn
from magma.utils.checkpoint import CheckpointManager, get_checkpoint_manager
import wandb


class BaseSolver(ABC, flashy.BaseSolver):
    """
    Base solver for SSSE
    Args:
        cfg (DictConfig): Configuration.
    """
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.logger.info(f"Instantiating solver {self.__class__.__name__} for XP {self.xp.sig}")
        self.logger.info(f"All XP logs are stored in {self.xp.folder}")
        self.log_cfg_summary(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.model: nn.Module
        self.dataloaders: tp.Dict[str, torch.utils.data.DataLoader] = dict()
        self.checkpoints: CheckpointManager
        self._log_updates = 10
        self._configure_checkpoints()
        self._configure_logging()
        self.build_dataloaders()
        self.build_model()
        self.build_optimizers()

    def log_cfg_summary(self, cfg: omegaconf.DictConfig, pref=""):
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                self.logger.info(f"{pref}{k}:")
                self.log_cfg_summary(v, pref + "  ")
            else:
                self.logger.info(f"{pref}{k}: {v}")

    def log_model_summary(self, model: nn.Module):
        """Log model summary, architecture and size of the model.
        """
        self.logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
        self.logger.info("Size: %.1f MB", mb)

    @abstractmethod
    def build_model(self):
        """Method to implement to initialize model.
        """
        ...

    @abstractmethod
    def build_dataloaders(self):
        """Method to implement to initialize dataloaders.
        """
        ...
    
    @abstractmethod
    def build_optimizers(self):
        """Method to implement to initialize dataloaders.
        """
        ...

    @abstractmethod
    def show(self):
        """Method to log any information without running the job.
        """
        ...

    @property
    def log_updates(self):
        # convenient access to log updates
        return self._log_updates

    @property
    def checkpoint_path(self):
        # we override the checkpoint_path to use the checkpoint manager
        return self.checkpoints.checkpoint_path

    def _configure_checkpoints(self):
        self.checkpoints = get_checkpoint_manager(self.logger, self.folder, self.cfg.solver.checkpoint)

    def _configure_logging(self):
        self._log_updates = self.cfg.logging.get('log_updates', self._log_updates)
        if self.cfg.logging.log_tensorboard:
            self.init_tensorboard(**self.cfg.get('tensorboard'))
        if self.cfg.logging.log_wandb and self:
            self.init_wandb(**self.cfg.get('wandb'))

    def load_checkpoints(self) -> bool:
        """Load last checkpoint or the one specified in continue_from.
        """
        state = self.checkpoints.load_checkpoints()
        if state is None:
            return False

        self.load_state_dict(state)
        return True

    def save_checkpoints(self):
        """Save checkpoint, optionally keeping a copy for a given epoch.
        """
        state = self.state_dict()
        self.checkpoints.save_checkpoints(state, self.epoch)

    def commit(self):
        """Commit metrics to dora and save checkpoints at the end of an epoch.
        """
        # We override commit to introduce more complex checkpoint saving behaviors
        if flashy.distrib.is_rank_zero():
            self.save_checkpoints()
        # This will increase self.epoch
        self.history.append(self._pending_metrics)

        # # log
        # if self.cfg.logging.log_wandb:
        #     tmp = {}
        #     for subset, metrics in self._pending_metrics.items():
        #         for k, v in metrics.items():
        #             tmp[f'{subset}_{k}'] = v

        #     wandb.log(tmp, step=self.epoch)

        self._start_epoch()
        if flashy.distrib.is_rank_zero():
            self.xp.link.update_history(self.history)

    def restore(self, replay_metrics: bool = True) -> bool:
        """Restore the status of a solver for a given experiment.
        """
        self.logger.info("Restoring weights and history")
        restored_checkpoints = self.load_checkpoints()
        if replay_metrics and len(self.history) > 0:
            self.logger.info("Replaying past metrics...")
            for epoch, stages in enumerate(self.history):
                for stage_name, metrics in stages.items():
                    # We manually log the metrics to the result logger
                    # as we don't want to add them to the pending metrics
                    self.result_logger.log_metrics(stage_name, metrics, step=epoch + 1, step_name='epoch',
                                                   formatter=self.get_formatter(stage_name))
        return restored_checkpoints

    def log_to_wandb(self):
        if self.cfg.logging.log_wandb:
            tmp = {}
            for subset, metrics in self._pending_metrics.items():
                for k, v in metrics.items():
                    tmp[f'{subset}_{k}'] = v

            wandb.log(tmp, step=self.epoch)


    def run(self):
        """Training loop.
        """
        assert len(self.state_dict()) > 0
        self.logger.info("Restoring checkpoint if such exist.")
        self.restore()  # load checkpoint and replay history
        self.logger.info("Training.")
        
        for epoch in range(self.epoch, self.cfg.solver.optim.epochs + 1):
            if self.should_stop_training():
                return

            # Stages are used for automatic metric reporting to Dora, and it also
            # allows tuning how metrics are formatted.
            metrics = self.run_stage('train', self.train)

            if self.should_run_stage('valid'):
                metrics = self.run_stage('valid', self.valid)

            if self.should_run_stage('evaluate'):
                metrics = self.run_stage('evaluate', self.evaluate)

            if self.should_run_stage('generate'):
                metrics = self.run_stage('generate', self.generate)

            # Commit will send the metrics to Dora and save checkpoints by default.
            self.commit()

    def should_stop_training(self) -> bool:
        """Check whether we should stop training or not.
        """
        return self.epoch > self.cfg.solver.optim.epochs

    def should_run_stage(self, stage_name) -> bool:
        """Check whether we want to run the specified stages.
        """
        stage_every = self.cfg.dset.eval_every[stage_name]
        is_last_epoch = self.epoch == self.cfg.solver.optim.epochs
        is_epoch_every = (stage_every and self.epoch % stage_every == 0)
        return is_last_epoch or is_epoch_every

    @abstractmethod
    def train(self):
        """Train stage.
        """
        ...

    @abstractmethod
    def valid(self):
        """Valid stage.
        """
        ...

    @abstractmethod
    def evaluate(self):
        """Evaluate stage.
        """
        ...

    @abstractmethod
    def generate(self):
        """Generate stage.
        """
        ...

    def run_generate(self):
        """Run only the evaluate stage.
        This method is useful to only generate samples from a trained experiment.
        """
        # TODO: Support generating from a given epoch?
        assert len(self.state_dict()) > 0
        self.restore(replay_metrics=False)  # load checkpoint and replay history
        self.run_stage('generate', self.generate)