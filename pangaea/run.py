import hashlib
import os as os
import pathlib
import pprint
import time
import numpy as np
import rasterio
from tqdm import tqdm

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)


def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info

@torch.no_grad()
def export_transfer_pred_tifs_sliding(
    *,
    model,                 # your DDP decoder
    loader: DataLoader,    # transfer_loader
    evaluator: Evaluator,  # SegEvaluator instance 
    out_dir: pathlib.Path,
    rank: int,
    logger,
):
    """
    Writes GeoTIFF prediction maps (*_pred.tif) with values 0..4

    Expects dataset batch:
      data["image"] is dict of modalities
      data["metadata"]["filename"] is the pre-image tif path (string or list of strings)
    """
    if rank != 0:
        return  # only rank0 writes files

    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    input_size = model.module.encoder.input_size  # 224

    for data in tqdm(loader, desc="Transfer export (pred tifs)"):
        image = data["image"]
        image = {k: v.to(evaluator.device, non_blocking=True) for k, v in image.items()}

        filenames = [m["filename"] for m in data["metadata"]]
        if isinstance(filenames, (str, pathlib.Path)):
            filenames = [str(filenames)]

        # EXACT same inference pathway as metrics (sliding + merge)
        logits = evaluator.sliding_inference(
            model,
            image,
            input_size=input_size,
            output_shape=None,  # keep native size (should be 1024x1024)
            max_batch=getattr(evaluator, "sliding_inference_batch", None),
        )

        # multi-class => 0..4
        pred = torch.argmax(logits, dim=1).to(torch.uint8)  # [B,H,W]

        for i, fn_pre in enumerate(filenames):
            src_path = pathlib.Path(fn_pre)

            with rasterio.open(src_path) as src:
                H, W = src.height, src.width

                arr = pred[i].detach().cpu().numpy()

                if arr.shape != (H, W):
                    arr_t = torch.from_numpy(arr)[None, None].float()
                    arr = torch.nn.functional.interpolate(
                        arr_t, size=(H, W), mode="nearest"
                    )[0, 0].byte().numpy()

                # CLEAN profile: keep only georeferencing + size
                out_profile = {
                    "driver": "GTiff",
                    "height": H,
                    "width": W,
                    "count": 1,
                    "dtype": rasterio.uint8,
                    "crs": src.crs,
                    "transform": src.transform,
                    "compress": "deflate",
                }

                out_path = out_dir / f"{src_path.stem}_pred.tif"
                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(arr, 1)

            logger.info(f"[transfer] wrote {out_path}")

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_info = get_exp_info(HydraConfig.get())
        exp_name = exp_info["exp_name"]
        task_name = exp_info["task"]
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        cfg_path = exp_dir / "configs" / "config.yaml"
        loaded_cfg = OmegaConf.load(cfg_path)
        cfg = OmegaConf.merge(loaded_cfg, cfg)
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    # prepare the decoder (segmentation/regression)
    decoder: Decoder = instantiate(
        cfg.decoder,
        encoder=encoder,
    )
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
        decoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.finetune,
    )
    logger.info(
        "Built {} for with {} encoder.".format(
            decoder.module.model_name, type(encoder).__name__
        )
    )

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training
    if train_run or cfg.task.trainer.model == "knn_probe":
        # get preprocessor
        train_preprocessor = instantiate(
            cfg.preprocessing.train,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        val_preprocessor = instantiate(
            cfg.preprocessing.val,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )

        # get datasets
        raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
        raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")

        if 0 < cfg.limited_label_train < 1:
            indices = get_subset_indices(
                raw_train_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_train,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_train_dataset = GeoFMSubset(raw_train_dataset, indices)

        if 0 < cfg.limited_label_val < 1:
            indices = get_subset_indices(
                raw_val_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_val,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_val_dataset = GeoFMSubset(raw_val_dataset, indices)

        train_dataset = GeoFMDataset(
            raw_train_dataset, train_preprocessor, cfg.data_replicate
        )
        val_dataset = GeoFMDataset(
            raw_val_dataset, val_preprocessor, cfg.data_replicate
        )

        logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

        logger.info(
            f"Total number of train patches: {len(train_dataset)}\n"
            f"Total number of validation patches: {len(val_dataset)}\n"
        )

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )

        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(cfg.optimizer, params=decoder.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )

        val_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device
        )
        trainer: Trainer = instantiate(
            cfg.task.trainer,
            model=decoder,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=val_evaluator,
            exp_dir=exp_dir,
            device=device,
        )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.ckpt_dir)

        trainer.train()

    
    # Evaluation
    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_evaluator: Evaluator = instantiate(
        cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device
    )

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
        
    if model_ckpt_path is None and not cfg.task.trainer.model_name == "knn_probe":
        raise ValueError(f"No model checkpoint found in {exp_dir}")
    
        # --- choose checkpoint path as you already do ---
    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)

    if model_ckpt_path is None and not cfg.task.trainer.model_name == "knn_probe":
        raise ValueError(f"No model checkpoint found in {exp_dir}")

    # --- normal test metrics (may fail if dataset has no target) ---
    # try:
    #     test_evaluator.evaluate(decoder, "test_model", model_ckpt_path)
    # except KeyError as e:
    #     logger.warning(f"Skipping test metrics (missing key in batch): {e}")

    # ------------------------------------------------------------
    # Transfer export
    # ------------------------------------------------------------
    if cfg.get("transfer", {}).get("enabled", False):
        out_dir = pathlib.Path(cfg.transfer.out_dir)

        # Build transfer dataset/loader
        transfer_preprocessor = instantiate(
            cfg.preprocessing.test,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        raw_transfer_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
        transfer_dataset = GeoFMDataset(raw_transfer_dataset, transfer_preprocessor)

        transfer_loader = DataLoader(
            transfer_dataset,
            sampler=DistributedSampler(transfer_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # ensure weights are loaded 
        model_dict = torch.load(model_ckpt_path, map_location=device, weights_only=False)
        if "model" in model_dict:
            decoder.module.load_state_dict(model_dict["model"])
        else:
            decoder.module.load_state_dict(model_dict)
        logger.info(f"[transfer] Loaded {model_ckpt_path} into decoder for export")

        export_transfer_pred_tifs_sliding(
            model=decoder,
            loader=transfer_loader,
            evaluator=test_evaluator,  
            out_dir=out_dir,
            rank=rank,
            logger=logger,
        )

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
