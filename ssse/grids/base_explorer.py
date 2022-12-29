import typing as tp
import treetable as tt
from abc import ABC, abstractmethod
from dora import Explorer


class BaseExplorer(ABC, Explorer):
    """Base explorer for MAGMA.

    All task specific solvers are expected to implement the `get_grid_metrics`
    method to specify logic about metrics to display for a given task.

    If additional stages are used, the child explorer must define how to handle
    these new stages in the `process_history` and `process_sheep` methods.
    """

    def stages(self):
        return ["train", "valid", "evaluate"]

    def get_grid_meta(self):
        """Returns the list of Meta information to display for each XP/job.
        """
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
        ]

    @abstractmethod
    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table.
        """
        ...

    def process_sheep(self, sheep, history):
        train = {
            "epoch": len(history),
        }
        parts = {"train": train}
        for metrics in history:
            for key, sub in metrics.items():
                part = parts.get(key, {})
                part.update(sub)
                parts[key] = part
        return parts

    def process_history(self, history):
        history_stages = {stage_name: {} for stage_name in self.stages}
        history_stages["train"] = {
            "epoch": len(history),
        }
        if history:
            metrics = history[-1]
            history_stages["train"].update(metrics["train"])

        for metrics in history:
            for stage_name in self.stages:
                if stage_name == "train":
                    continue
                elif stage_name in metrics:
                    history_stages[stage_name] = metrics[stage_name]

        return history_stages
