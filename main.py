import argparse
import yaml

from dataset import ReverseStringDataset, create_dataloader
from model import ReverseStringModel
from trainer import BaseTrainer
import torch


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a reverse string model.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
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
    config = load_yaml_config(args.config)

    # Override device if provided
    if args.device:
        config["trainer"]["device"] = args.device

    device = config["trainer"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"

    # Create datasets
    dataset_config = config["dataset"]
    dataset = ReverseStringDataset(
        min=dataset_config.get("min", 5),
        max=dataset_config.get("max", 100),
        size=dataset_config.get("size", 10000),
        fixed_length=dataset_config.get("fixed_length"),
    )

    train_loader, val_loader = create_dataloader(
        dataset=dataset,
        batch_size=dataset_config.get("batch_size", 64),
        shuffle=dataset_config.get("shuffle", True),
        train_split=dataset_config.get("train_split", 0.8),
    )

    # Create model
    model_config = config.get("model", {})
    model = ReverseStringModel(
        num_layers=model_config.get("num_layers", 2),
        embed_dim=model_config.get("embed_dim", 128),
        intermediate=model_config.get("intermediate", 512),
        heads=model_config.get("heads", 4),
        vocab_size=len(dataset.char_to_idx),
        pad_idx=dataset.char_to_idx.get("<PAD>"),
    ).to(device)

    print(f"Model: {model}")
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
