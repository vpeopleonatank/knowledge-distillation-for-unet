from typing import List
import collections
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import numpy as np
from SegmentationDataset import SegmentationDataset
from pathlib import Path

def get_loaders(
    images: List[Path],
    masks: List[Path],
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transforms_fn = None,
    valid_transforms_fn = None,
    to_mask2d = False,
    return_dict=True,
) -> dict:
 
    indices = np.arange(len(images))
 
    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
      indices, test_size=valid_size, random_state=random_state, shuffle=True
    )
 
    np_images = np.array(images)
    np_masks = np.array(masks)
 
    # Creates our train dataset
    train_dataset = SegmentationDataset(
      images = np_images[train_indices].tolist(),
      masks = np_masks[train_indices].tolist(),
      transforms = train_transforms_fn,
      to_mask2d = to_mask2d,
      return_dict = return_dict,
    )
 
    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
      images = np_images[valid_indices].tolist(),
      masks = np_masks[valid_indices].tolist(),
      transforms = valid_transforms_fn,
      to_mask2d = to_mask2d,
      return_dict = return_dict,
    )
 
 
    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      drop_last=True,
    )
 
    valid_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      drop_last=True,
    )
 
    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader
 
    return loaders
