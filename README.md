# reverse-string-model

Train a decoder-only Transformer to reverse randomly generated strings using a fully YAML-driven training pipeline.

## Highlights

- Config-first training (`config.yaml`) for dataset, model, optimizer, scheduler, loss, metrics, trainer, and handlers.
- Trainer-centric handler system: custom handlers receive the `trainer` object and can access model, loaders, evaluators, optimizer, scheduler, and engine.
- Ignite-based training loop with optional evaluation, checkpointing, early stopping, and TensorBoard logging.

## Project structure

```text
.
├── config.yaml         # Full training/eval/handler configuration
├── dataset.py          # Synthetic reverse-string dataset + dataloader utilities
├── handler.py          # Custom Ignite handlers (checkpoints, early stopping, TB, logging)
├── main.py             # Entry point: load config, build objects, run training
├── model.py            # Decoder-only Transformer for sequence generation
├── train.py            # Train/eval step functions, loss and metric factories
├── trainer.py          # BaseTrainer with dynamic training loop and handler integration
```

## Requirements

- Python `>=3.12`
- [uv](https://github.com/astral-sh/uv) (recommended)

Install:

```bash
uv sync
```

## Run training

Use default config:

```bash
uv run python main.py
```

Use a custom config file:

```bash
uv run python main.py --config path/to/config.yaml
```

Override device from CLI:

```bash
uv run python main.py --device cpu
```

If `trainer.device` is `cuda` and CUDA is unavailable, training automatically falls back to CPU.

## Configuration guide

All behavior is controlled by `config.yaml`.

### 1) Core sections

- `dataset`: synthetic data generation and dataloader split/batch settings.
- `model`: Transformer dimensions (`num_layers`, `embed_dim`, `intermediate`, `heads`).
- `optimizer` / `scheduler`: fully dynamic from import paths.
- `loss_fn` / `metrics`: loaded from factory function paths.
- `trainer`: training/eval step function paths and run args.

Example dynamic function path:

```yaml
trainer:
  train_step: train.train_step
  eval_step: train.eval_step
```

### 2) Handlers

Handlers are configured via:

```yaml
handlers:
  path: handler.attach_handlers
  kargs: ...
```

`BaseTrainer` calls `handlers.path` with `trainer` as the first argument and `handlers.kargs` as keyword arguments.

Inside `handler.attach_handlers(trainer, **config)`, you can directly access:

- `trainer.engine`
- `trainer.model`
- `trainer.optimizer`
- `trainer.scheduler`
- `trainer.train_loader` / `trainer.val_loader`
- `trainer.train_evaluator` / `trainer.val_evaluator`

Current handler config supports:

- Iteration loss logging (`log_every`)
- Epoch metric logging (`log_metrics`)
- Multi-policy checkpointing (`checkpoints`)
- Early stopping (`early_stopping`)
- TensorBoard output + optimizer params (`tensorboard`)

## How training is wired

1. `main.py` loads YAML and builds dataset/loaders/model.
2. `BaseTrainer` builds optimizer/scheduler/loss/train step/eval step.
3. Evaluators are created when `trainer.evaluation: true`.
4. Custom handlers are attached via `handlers.path`.
5. `trainer.run()` starts Ignite training.

## Checkpoints and TensorBoard

- Checkpoints are written to `checkpoint/` based on handler policies (`best`, `latest`, `periodic`).
- TensorBoard logs are written to `tb-logs/`.

Launch TensorBoard:

```bash
uv run tensorboard --logdir tb-logs
```
