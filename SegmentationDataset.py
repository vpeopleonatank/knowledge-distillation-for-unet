from typing import List

from torch.utils.data import Dataset
from skimage import io
import cv2
from pathlib import Path
from skimage.io import imread


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: List[Path],
        masks: List[Path] = None,
        transforms=None,
        to_mask2d=False
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.to_mask2d = to_mask2d

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = io.imread(image_path)

        result = {"image": image}

        if self.masks is not None:
            mask = imread(self.masks[idx])
            if self.to_mask2d:
              mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result
