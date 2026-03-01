import argparse
import yaml
import os
import shutil
from datetime import datetime
from pathlib import Path

from src.dataset import create_dataloader
from src.trainer import BaseTrainer, parse_function_from_string
import torch


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reverse string model.")
    parser.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        help="Path to the YAML configuration file (positional).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory to save logs, configs, and checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train the model on (overrides config).",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_path = args.config_pos if args.config_pos else args.config
    config = load_yaml_config(config_path)

    # Setup output directory
    output_dir = None
    if args.output:
        config_name = Path(config_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output) / config_name / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy configuration
        shutil.copy(config_path, output_dir / f"{config_name}.yaml")

        print(f"Output directory initialized at {output_dir}")

        # Override paths in handlers to point to output_dir
        if "handlers" in config and "kargs" in config["handlers"]:
            h_cfg = config["handlers"]["kargs"]
            if "checkpoints" in h_cfg:
                for ckpt in h_cfg["checkpoints"].values():
                    # keep dirname relative to output_dir or absolute
                    dirname = ckpt.get("dirname", "checkpoint")
                    ckpt["dirname"] = str(output_dir / dirname)
            if "tensorboard" in h_cfg:
                log_dir = h_cfg["tensorboard"].get("log_dir", "tb-logs")
                h_cfg["tensorboard"]["log_dir"] = str(output_dir / "tensorboard")

    # Override device if provided
    if args.device:
        config["trainer"]["device"] = args.device

    device = config["trainer"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        device = "cpu"

    # Create datasets
    dataset_config = config["dataset"]
    dataset = parse_function_from_string(
        dataset_config.get("path", "dataset.ReverseStringDataset")
    )(**dataset_config.get("kargs", {}))

    train_loader, val_loader = create_dataloader(
        dataset=dataset,
        **dataset_config.get("dataloader_kargs", {"batch_size": 32, "shuffle": True}),
    )
    # Create model
    model_config = config.get("model", {})
    model_path = model_config.get("path", "model.ReverseStringModel")
    model = parse_function_from_string(model_path)(**model_config.get("kargs", {})).to(
        device
    )

    print(f"Model:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = BaseTrainer(
        model=model,
        device=device,
        cfg=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Start training
    trainer.run()

    # Clean up handlers
    tb_logger = trainer.handlers.get("tensorboard")
    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    main()
