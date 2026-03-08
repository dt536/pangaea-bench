import re
from typing import Sequence
import numpy as np
import torch
import os
import rasterio
import cv2

from pangaea.datasets.base import RawGeoFMDataset


class damage_w_masks(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        oversample_building_damage: bool
    ):
        """Initialize the earthquake dataset

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image. 
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality. 
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            oversample_building_damage (bool): whether to oversample images with building damage
        """
        super(damage_w_masks, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        self.root_path = root_path
        self.split = split
        self.bands = bands
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download
        self.oversample_building_damage = oversample_building_damage

        self.all_files = self.get_all_files()


    def get_all_files(self) -> Sequence[str]:
        split_dir = os.path.join(self.root_path, self.split)
        pre_dir = os.path.join(split_dir, "pre")

        if not os.path.exists(pre_dir):
            raise FileNotFoundError(f"Missing folder: {pre_dir}")

        all_files = [
            os.path.join(pre_dir, f)
            for f in sorted(os.listdir(pre_dir))
            if re.match(r"^pre_r\d+_c\d+\.tif$", f, flags=re.IGNORECASE)
        ]

        if len(all_files) == 0:
            raise RuntimeError(f"No pre_rX_cY.tif files found in {pre_dir}")

        return all_files


    def __len__(self) -> int:
        return len(self.all_files)

    @staticmethod
    def read_tif_rgb(path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            x = src.read(out_dtype=np.float32)  # (C,H,W)
        x = np.transpose(x, (1, 2, 0))          # (H,W,C)
        return x

    def __getitem__(self, idx: int):
        fn_pre = self.all_files[idx]

        fn_post = fn_pre.replace(os.sep + "pre" + os.sep, os.sep + "post" + os.sep)
        fn_post = os.path.join(
            os.path.dirname(fn_post),
            os.path.basename(fn_post).replace("pre_", "post_", 1),
        )

        fn_mask = fn_pre.replace(os.sep + "pre" + os.sep, os.sep + "target" + os.sep)
        fn_mask = os.path.join(
            os.path.dirname(fn_mask),
            os.path.basename(fn_mask).replace("pre_", "target_", 1).replace(".tif", ".png"),
        )

        if not os.path.exists(fn_post):
            raise FileNotFoundError(f"Missing post image:\n{fn_post}")
        if not os.path.exists(fn_mask):
            raise FileNotFoundError(f"Missing target mask:\n{fn_mask}")

        img_pre = self.read_tif_rgb(fn_pre)
        img_post = self.read_tif_rgb(fn_post)

        img_pre = img_pre[..., ::-1].copy()
        img_post = img_post[..., ::-1].copy()

        msk = cv2.imread(fn_mask, cv2.IMREAD_UNCHANGED)
        if msk is None:
            raise RuntimeError(f"cv2 failed to read mask:\n{fn_mask}")

        if msk.ndim != 2:
            raise ValueError(f"Mask must be single-channel, got shape {msk.shape} for {fn_mask}")

        img = np.stack([img_pre, img_post], axis=0)              # (2,H,W,3)
        img = torch.from_numpy(img).permute(3, 0, 1, 2).float()  # (3,2,H,W)
        msk = torch.from_numpy(msk).long()

        return {
            "image": {"optical": img},
            "target": msk,
            "metadata": {"filename": fn_pre},
        }

if __name__=="__main__":
    print("Testing earthquake dataset loading...")