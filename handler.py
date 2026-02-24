from ignite import metrics
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler
import importlib
import torch



def get_metrics():
    return {
        "accuracy": metrics.Accuracy(output_transform=accuracy_transform),
        "loss": metrics.Loss(loss_fn),
    }


def _parse_event(event_cfg):
    if isinstance(event_cfg, dict):
        name = event_cfg.get("name", "EPOCH_COMPLETED")
        every = event_cfg.get("every")
        event = getattr(Events, name, Events.EPOCH_COMPLETED)
        if every is not None:
            event = event(every=int(every))
        return event
    if isinstance(event_cfg, str):
        return getattr(Events, event_cfg, Events.EPOCH_COMPLETED)
    return Events.EPOCH_COMPLETED


def _parse_function(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected string, got {type(value).__name__}")
    if value.startswith("lambda"):
        return eval(value, {"__builtins__": {}}, {})
    if "." in value:
        module_name, func_name = value.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    raise ValueError(f"Invalid function string: {value}")


def _resolve_to_save(keys, context):
    if not keys:
        return {}
    return {key: context[key] for key in keys if context.get(key) is not None}



def attach_handlers(
    trainer,
    train_evaluator,
    val_evaluator,
    train_loader,
    val_loader,
    config,
    context,
):
    handlers = {}

    log_every = config.get("log_every")
    if log_every:
        log_event = Events.ITERATION_COMPLETED(every=int(log_every))

        def _log_loss(engine):
            print(
                f"Epoch[{engine.state.epoch}] Iter[{engine.state.iteration}] "
                f"Loss: {engine.state.output:.4f}"
            )

        trainer.engine.add_event_handler(log_event, _log_loss)

    if config.get("log_metrics", True) and train_evaluator is not None:

        def _log_metrics(engine):
            train_evaluator.run(train_loader)
            train_metrics = train_evaluator.state.metrics
            train_str = f"Training Results - Epoch: {engine.state.epoch}"
            for name, value in train_metrics.items():
                train_str += f" {name}: {value:.4f}"
            print(train_str)

            if val_loader is not None and val_evaluator is not None:
                val_evaluator.run(val_loader)
                val_metrics = val_evaluator.state.metrics
                val_str = f"Validation Results - Epoch: {engine.state.epoch}"
                for name, value in val_metrics.items():
                    val_str += f" {name}: {value:.4f}"
                print(val_str)

        trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)

    checkpoints_cfg = config.get("checkpoints", {})
    for name, ckpt_cfg in checkpoints_cfg.items():
        if not ckpt_cfg.get("enabled", True):
            continue
        event = _parse_event(ckpt_cfg.get("event", "EPOCH_COMPLETED"))
        to_save = _resolve_to_save(ckpt_cfg.get("to_save", ["model"]), context)
        if not to_save:
            continue
        checkpoint = ModelCheckpoint(
            ckpt_cfg.get("dirname", "checkpoint"),
            n_saved=int(ckpt_cfg.get("n_saved", 1)),
            filename_prefix=ckpt_cfg.get("filename_prefix", name),
            score_function=_parse_function(ckpt_cfg.get("score_function")),
            score_name=ckpt_cfg.get("score_name"),
            global_step_transform=_parse_function(
                ckpt_cfg.get("global_step_transform")
            ),
        )
        target = ckpt_cfg.get("target", "trainer")
        if target == "val_evaluator" and val_evaluator is not None:
            val_evaluator.add_event_handler(event, checkpoint, to_save)
        else:
            trainer.engine.add_event_handler(event, checkpoint, to_save)
        handlers[name] = checkpoint

    early_stopping_cfg = config.get("early_stopping", {})
    if early_stopping_cfg.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=int(early_stopping_cfg.get("patience", 3)),
            score_function=_parse_function(early_stopping_cfg.get("score_function")),
            trainer=trainer.engine,
            min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
        )
        event = _parse_event(early_stopping_cfg.get("event", "COMPLETED"))
        target = early_stopping_cfg.get("target", "val_evaluator")
        if target == "trainer":
            trainer.engine.add_event_handler(event, early_stopping)
        elif val_evaluator is not None:
            val_evaluator.add_event_handler(event, early_stopping)
        handlers["early_stopping"] = early_stopping

    tb_cfg = config.get("tensorboard", {})
    if tb_cfg.get("enabled", False):
        tb_logger = TensorboardLogger(log_dir=tb_cfg.get("log_dir", "./tb-logs"))
        tb_logger.attach_output_handler(
            trainer.engine,
            event_name=Events.ITERATION_COMPLETED,
            tag=tb_cfg.get("batch_loss_tag", "training"),
            output_transform=lambda loss: {"batch_loss": loss},
        )

        metric_names = tb_cfg.get("metric_names", ["accuracy", "loss"])
        global_step_transform = _parse_function(tb_cfg.get("global_step_transform"))

        if train_evaluator is not None:
            tb_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=metric_names,
                global_step_transform=global_step_transform,
            )
        if val_evaluator is not None:
            tb_logger.attach_output_handler(
                val_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=metric_names,
                global_step_transform=global_step_transform,
            )

        if tb_cfg.get("log_optimizer_params", True) and context.get("optimizer"):
            tb_logger.attach(
                trainer.engine,
                log_handler=OptimizerParamsHandler(context["optimizer"]),
                event_name=Events.ITERATION_STARTED,
            )
        handlers["tensorboard"] = tb_logger

    return handlers
