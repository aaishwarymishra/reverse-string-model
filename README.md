# reverse-string-model

A PyTorch-based transformer model that learns to reverse strings using attention mechanisms.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- Python 3.12 or higher

## Installation

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/aaishwarymishra/reverse-string-model.git
cd reverse-string-model
```

3. Install dependencies using `uv`:
```bash
uv sync
```

## Running the Script

### Basic Usage

Run the training script with default parameters:
```bash
uv run python main.py
```

### Custom Configuration

You can customize the training by passing command-line arguments:

```bash
uv run python main.py --batch_size 64 --num_epochs 20 --learning_rate 0.001
```

### Available Arguments

- `--batch_size`: Batch size for training (default: 64)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate for the optimizer (default: 1e-3)
- `--max_length`: Maximum length of input strings (default: 100)
- `--fixed_length`: Fixed length for input strings (default: 100)
- `--scheduler`: Enable learning rate scheduler (flag)
- `--num_layers`: Number of transformer layers (default: 2)
- `--embed_dim`: Embedding dimension (default: 128)
- `--intermediate`: Intermediate dimension for feedforward network (default: 512)
- `--heads`: Number of attention heads (default: 4)
- `--train_size`: Number of training samples (default: 10000)
- `--val_size`: Number of validation samples (default: 1000)
- `--device`: Device to train on - 'cpu' or 'cuda' (default: cuda)

### Example with Custom Parameters

```bash
uv run python main.py \
  --batch_size 128 \
  --num_epochs 15 \
  --learning_rate 0.0005 \
  --num_layers 4 \
  --embed_dim 256 \
  --heads 8 \
  --device cuda \
  --scheduler
```

## Project Structure

- `main.py` - Main training script and configuration
- `model.py` - Transformer model implementation
- `dataset.py` - Dataset generation and data loading
- `trainer.py` - Training loop and validation logic
- `pyproject.toml` - Project dependencies and metadata

## Dependencies

- PyTorch
- pytorch-ignite (>=0.5.3)
- numpy (>=2.4.2)
- pydantic (>=2.12.5)