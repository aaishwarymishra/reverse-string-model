from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import LRScheduler
from typing import Callable, Any
import torch
import importlib


def parse_function_from_string(val: str):
    if not isinstance(val, str):
        raise TypeError(f"Expected string, got {type(val).__name__}")
    if val.startswith("lambda"):
        return eval(val, {"__builtins__": None}, {})
    elif "." in val:
        module_name, func_name = val.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    else:
        raise ValueError(f"Invalid function string: {val}")


class BaseTrainer:
    def __init__(self, model, cfg, train_loader, val_loader=None, device="cpu"):
        self.model = model
        self.device = device
        self.cfg: dict = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_evaluator = None
        self.val_evaluator = None
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_step = self.get_train_step()
        self.eval_step = self.get_eval_step()
        self.engine = Engine(self.train_step)
        self.loss_fn = self.get_loss_fn()

        if self.cfg.get("trainer", {}).get("evaluation", False):
            self.train_evaluator, self.val_evaluator = self.create_evaluators()
        self.handlers = self.attach_handlers()
        self.attach_scheduler()

    def get_train_step(self) -> Callable:
        """Returns the training step function, either from config or default."""
        if self.cfg.get("trainer", {}).get("train_step") is not None:
            train_step = parse_function_from_string(
                self.cfg.get("trainer", {}).get("train_step")
            )

            def train_step_wrapper(engine, batch):
                return train_step(
                    engine, self.model, batch, self.loss_fn, self.optimizer, self.device
                )

            return train_step_wrapper
        else:
            return self.default_train_step

    def default_train_step(self):
        """Default training step function."""
        raise NotImplementedError(
            "train_step method must be implemented in subclass or provided via config"
        )

    def default_optimizer(self) -> torch.optim.Optimizer:
        """Returns the default optimizer, must be implemented in subclass."""
        raise NotImplementedError(
            "build_optimizer method must be implemented in subclass or provided via config"
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config or return default."""
        optimizer_cfg = self.cfg.get("optimizer")
        if optimizer_cfg is not None and optimizer_cfg.get("path") is not None:
            optimizer = parse_function_from_string(optimizer_cfg.get("path"))
            return optimizer(self.model.parameters(), **optimizer_cfg.get("kargs", {}))
        else:
            return self.default_optimizer()

    def default_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Returns the default scheduler."""
        return None

    def build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build scheduler from config or return default."""
        scheduler_cfg = self.cfg.get("scheduler")
        if scheduler_cfg is not None and scheduler_cfg.get("path") is not None:
            scheduler = parse_function_from_string(scheduler_cfg.get("path"))
            return scheduler(self.optimizer, **scheduler_cfg.get("kargs", {}))
        else:
            return self.default_scheduler()

    def attach_scheduler(self):
        """Attach scheduler to engine events if configured."""
        if self.scheduler is not None:
            scheduler_cfg = self.cfg.get("scheduler", {})
            event = scheduler_cfg.get("event", "ITERATION_COMPLETED")

            if isinstance(event, str):
                event = getattr(Events, event, Events.ITERATION_COMPLETED)
            elif isinstance(event, dict) and "every" in event and "name" in event:
                event = getattr(
                    Events,
                    event.get("name", "ITERATION_COMPLETED"),
                    Events.ITERATION_COMPLETED,
                )(every=int(event["every"]))

            self.scheduler_handler = LRScheduler(
                self.scheduler, **scheduler_cfg.get("handler_kargs", {})
            )
            self.engine.add_event_handler(event, self.scheduler_handler)

    def default_metrics(self) -> dict[str, Any]:
        """Returns the default metrics dict for evaluators."""
        raise NotImplementedError(
            "get_metrics method must be implemented in subclass or provided via config"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Metrics for evaluators (attached during evaluation)."""
        metrics_cfg = self.cfg.get("metrics")
        if metrics_cfg is not None and metrics_cfg.get("path") is not None:
            metrics = parse_function_from_string(metrics_cfg.get("path"))(
                self.loss_fn, **metrics_cfg.get("kargs", {})
            )
            return metrics
        return self.default_metrics()

    def default_eval_step(self):
        """The default evaluation step function."""
        raise NotImplementedError(
            "eval_step method must be implemented in subclass or provided via config"
        )

    def get_eval_step(self) -> Callable:
        """Returns the evaluation step function, either from config or default."""
        if self.cfg.get("trainer", {}).get("eval_step") is not None:
            eval_step = parse_function_from_string(
                self.cfg.get("trainer", {}).get("eval_step")
            )

            def eval_step_wrapper(engine, batch):
                return eval_step(engine, self.model, batch, self.device)

            return eval_step_wrapper
        else:
            return self.default_eval_step

    def create_evaluators(self) -> tuple[Engine, Engine] | tuple[None, None]:
        """Create train and validation evaluators with metrics."""
        eval_metrics = self.get_metrics()
        if eval_metrics is None:
            return None, None

        train_evaluator = Engine(self.eval_step)
        val_evaluator = Engine(self.eval_step)

        train_metrics = self.get_metrics()
        val_metrics = self.get_metrics()

        if train_metrics:
            for name, metric in train_metrics.items():
                metric.attach(train_evaluator, name)

        if val_metrics:
            for name, metric in val_metrics.items():
                metric.attach(val_evaluator, name)

        return train_evaluator, val_evaluator

    def default_loss_fn(self):
        raise NotImplementedError(
            "get_loss_fn method must be implemented in subclass or provided via config"
        )

    def get_loss_fn(self):
        loss_fn_cfg = self.cfg.get("loss_fn")
        if loss_fn_cfg is not None and loss_fn_cfg.get("path") is not None:
            loss_fn = parse_function_from_string(loss_fn_cfg.get("path"))(
                **loss_fn_cfg.get("kargs", {})
            )
            return loss_fn
        else:
            return self.default_loss_fn()
        
    def default_handlers(self):
        """Default handlers to attach, can be overridden or provided via config."""
        return {}

    def attach_handlers(self):
        handlers = self.default_handlers()
        if self.cfg.get("handlers"):
            handlers_cfg = self.cfg.get("handlers", {})
            path = handlers_cfg.get("path")
            if path is not None:
                handler_args = dict(handlers_cfg.get("kargs", {}))
                for key, value in handlers_cfg.items():
                    if key not in {"path", "kargs"} and key not in handler_args:
                        handler_args[key] = value
                custom_handlers = parse_function_from_string(path)(self, **handler_args)
                if custom_handlers:
                    handlers.update(custom_handlers)
        return handlers

    def run(self, train_loader=None, val_loader=None):
        """Run training with optional validation."""
        train_loader = train_loader or self.train_loader
        val_loader = val_loader or self.val_loader
        trainer_cfg = self.cfg.get("trainer", {})
        run_kargs = trainer_cfg.get("run_kargs", {"max_epochs": 10})
        self.engine.run(train_loader, **run_kargs)

    @staticmethod
    def default_evaluation_log_fn(
        engine,
        train_evaluator,
        val_evaluator=None,
        train_loader=None,
        val_loader=None,
    ):
        """Default evaluation logging function, can be overridden or provided via config."""
        raise NotImplementedError(
            "default_evaluation_log_fn method must be implemented in subclass or provided via config"
        )
