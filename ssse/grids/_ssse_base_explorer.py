import typing as tp
import treetable as tt

from ._base_explorer import BaseExplorer

class SsseBaseExplorer(BaseExplorer):
    eval_metrics: tp.List[str] = []

    def stages(self) -> tp.List[str]:
        return ["train", "valid"]

    def get_grid_meta(self):
        """Returns the list of Meta information to display for each XP/job.
        """
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
        ]

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table.
        """
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("tot_loss", ".4f"),  # total loss
                    tt.leaf("reconst", ".4f"),  # total loss
                    tt.leaf("cont", ".5f"),  # total loss
                    tt.leaf("reg", ".3f"),  # total loss
                    tt.leaf("energy", ".3f"),  # total loss
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("tot_loss", ".4f"),  # total loss
                    tt.leaf("reconst", ".4f"),  # total loss
                    tt.leaf("cont", ".5f"),  # total loss
                    tt.leaf("reg", ".3f"),  # total loss
                    tt.leaf("energy", ".3f"),  # total loss
                ],
                align=">",
            ),
            tt.group(
                "evaluate",
                [
                    tt.leaf("snr", ".2f"),
                    tt.leaf("pesq", ".2f"),
                    tt.leaf("stoi", ".2f"),
                ],
                align=">",
            ),
        ]
