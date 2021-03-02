from pathlib import Path

from cellpose import models
import torch
from dataloader import LoadImageBatch
import analysis

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class RunCellpose:
    def __init__(
        self,
        parent_folder=None,
        metadata_path=None,
        feature_to_segment=None,
        nucleus_channels=None,
        cyto_channels=None,
        estimated_diameter=None,
    ):

        self.parent_folder = parent_folder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8000000000 else "cpu")

        self.gpu = True if "cuda" in str(self.device.type) else False
        self.metadata_path = metadata_path
        self.feature_to_segment = feature_to_segment
        self.nucleus_channels = nucleus_channels
        self.cyto_channels = cyto_channels
        self.estimated_diameter = estimated_diameter
        # self.output_parent_path = Path.joinpath(Path(parent_folder),' Segmented')

        if self.feature_to_segment not in ("cyto", "nuclei", "both"):
            raise ValueError('No mode selected. Must enter "cyto" or "nuclei"')

        self.model = models.Cellpose(
            model_type=self.feature_to_segment,
            device=self.device,
            gpu=self.gpu,
            torch=True,
        )

        print("Finding images...")

        self.dataset = LoadImageBatch(
            self.parent_folder, self.metadata_path, self.device
        )

        print("Loaded.")

    def segment_images(self):

        # if not self.output_parent_path.exists():
        #    Path.mkdir(self.output_parent_path)

        print("Selecting batch...")

        for i in range(len(self.dataset)):
            batch = self.dataset[i]
            images = [set[0] for set in batch]
            metadata = [set[1] for set in batch]
            img_paths = [set[2] for set in batch]

            analysis.run_segmentation(
                self.parent_folder,
                images,
                metadata,
                img_paths,
                self.cyto_channels,
                self.nucleus_channels,
                self.estimated_diameter,
                self.model,
                self.feature_to_segment,
            )
