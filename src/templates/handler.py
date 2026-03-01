from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.handlers.tensorboard_logger import (
    TensorboardLogger,
    OptimizerParamsHandler,
)
from src.trainer import parse_function_from_string, _parse_event
import logging


def _resolve_to_save(keys, context):
    if not keys:
        return {}
    return {key: context[key] for key in keys if context.get(key) is not None}


def attach_handlers(trainer, **config):
    handlers = {}

    if not config:
        handlers_cfg = trainer.cfg.get("handlers", {})
        config = dict(handlers_cfg.get("kargs", {}))
        for key, value in handlers_cfg.items():
            if key not in {"path", "kargs"} and key not in config:
                config[key] = value

    train_evaluator = getattr(trainer, "train_evaluator", None)
    val_evaluator = getattr(trainer, "val_evaluator", None)
    train_loader = getattr(trainer, "train_loader", None)
    val_loader = getattr(trainer, "val_loader", None)
    context = {
        "model": getattr(trainer, "model", None),
        "optimizer": getattr(trainer, "optimizer", None),
        "scheduler": getattr(trainer, "scheduler", None),
    }

    log_every = config.get("log_every")
    if log_every:
        log_event = Events.ITERATION_COMPLETED(every=int(log_every))

        def _log_loss(engine):
            logging.info(
                f"Epoch[{engine.state.epoch}] Iter[{engine.state.iteration}] "
                f"Loss: {engine.state.output:.4f}"
            )

        trainer.engine.add_event_handler(log_event, _log_loss)

    if config.get("log_metrics", True) and train_evaluator is not None:

        def _log_metrics(engine):
            if train_loader is None:
                return
            train_evaluator.run(train_loader)
            train_metrics = train_evaluator.state.metrics
            train_str = f"Training Results - Epoch: {engine.state.epoch}"
            for name, value in train_metrics.items():
                train_str += f" {name}: {value:.4f}"
            logging.info(train_str)

            if val_loader is not None and val_evaluator is not None:
                val_evaluator.run(val_loader)
                val_metrics = val_evaluator.state.metrics
                val_str = f"Validation Results - Epoch: {engine.state.epoch}"
                for name, value in val_metrics.items():
                    val_str += f" {name}: {value:.4f}"
                logging.info(val_str)

        trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, _log_metrics)

    checkpoints_cfg = config.get("checkpoints", {})
    for name, ckpt_cfg in checkpoints_cfg.items():
        if not ckpt_cfg.get("enabled", True):
            continue
        event = _parse_event(ckpt_cfg.get("event", "EPOCH_COMPLETED"))
        to_save = _resolve_to_save(ckpt_cfg.get("to_save", ["model"]), context)
        if not to_save:
            continue
        score_func_str = ckpt_cfg.get("score_function")
        global_step_str = ckpt_cfg.get("global_step_transform")

        checkpoint = ModelCheckpoint(
            ckpt_cfg.get("dirname", "checkpoint"),
            n_saved=int(ckpt_cfg.get("n_saved", 1)),
            filename_prefix=ckpt_cfg.get("filename_prefix", name),
            score_function=parse_function_from_string(score_func_str)
            if score_func_str
            else None,
            score_name=ckpt_cfg.get("score_name"),
            global_step_transform=parse_function_from_string(global_step_str)
            if global_step_str
            else None,
        )
        target = ckpt_cfg.get("target", "trainer")
        if target == "val_evaluator" and val_evaluator is not None:
            val_evaluator.add_event_handler(event, checkpoint, to_save)
        else:
            trainer.engine.add_event_handler(event, checkpoint, to_save)
        handlers[name] = checkpoint

    early_stopping_cfg = config.get("early_stopping", {})
    if early_stopping_cfg.get("enabled", False):
        score_function = parse_function_from_string(
            early_stopping_cfg.get("score_function")
        )
        if score_function is None:
            raise ValueError("early_stopping.score_function must be provided")
        early_stopping = EarlyStopping(
            patience=int(early_stopping_cfg.get("patience", 3)),
            score_function=score_function,
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
        global_step_str_tb = tb_cfg.get("global_step_transform")
        global_step_transform = (
            parse_function_from_string(global_step_str_tb)
            if global_step_str_tb
            else None
        )

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

        optimizer = context.get("optimizer")
        if tb_cfg.get("log_optimizer_params", True) and optimizer is not None:
            tb_logger.attach(
                trainer.engine,
                log_handler=OptimizerParamsHandler(optimizer),
                event_name=Events.ITERATION_STARTED,
            )
        handlers["tensorboard"] = tb_logger

    return handlers
