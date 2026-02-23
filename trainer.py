from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import LRScheduler
import torch
import importlib


def parse_function_from_string(val: str):
    if not isinstance(val, str):
        raise TypeError(f"Expected string, got {type(val).__name__}")
    if val.startswith('lambda'):
        return eval(val, {"__builtins__": None}, {})
    elif '.' in val:
        module_name, func_name = val.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    else:
        raise ValueError(f"Invalid function string: {val}")


class BaseTrainer:
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg: dict = cfg
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_step = self.get_train_step()
        self.engine = Engine(self.train_step)
        self.loss_fn = self.get_loss_fn()


        self._train_evaluator = None
        self._val_evaluator = None

        self.attach_scheduler()
        self.attach_handlers()

    def get_train_step(self):
        return self.train_step

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch[0].to(self.device), batch[1].to(self.device)

        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def build_optimizer(self):
        return None

    def build_scheduler(self):
        return None

    def attach_scheduler(self):
        """Attach scheduler to engine events if configured."""
        if self.scheduler is not None:
            scheduler_cfg = self.cfg.get('scheduler', {})
            event = scheduler_cfg.get('event', 'ITERATION_COMPLETED')

            if isinstance(event, str):
                event = getattr(Events, event, Events.ITERATION_COMPLETED)
            elif isinstance(event, dict) and 'every' in event and 'name' in event:
                event = getattr(Events, event.get(
                    'name', 'ITERATION_COMPLETED'), Events.ITERATION_COMPLETED)(event['every'])

            self.scheduler_handler = LRScheduler(
                self.scheduler, **scheduler_cfg.get('handler_args', []), **scheduler_cfg.get('kwargs', {}))
            self.engine.add_event_handler(event, self.scheduler_handler)

    def get_metrics(self):
        return None

    def eval_step(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            y_pred = self.model(x)
            return y_pred, y

    def create_evaluators(self):
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

    def attach_evaluation_handler(self, config, train_loader, val_loader=None):
        """Attach evaluation handler with closure over dataloaders."""
        path = config.get('path')


        if path is not None:
            eval_handler_fn = parse_function_from_string(path)
        else:
            eval_handler_fn = self.log_evaluation

        # Wrapper to inject evaluators and dataloaders via closure
        def eval_handler_fn_wrapper(engine):
            return eval_handler_fn(
                engine,
                train_evaluator=self._train_evaluator,
                val_evaluator=self._val_evaluator,
                train_loader=train_loader,
                val_loader=val_loader
            )

        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED, eval_handler_fn_wrapper)

    def get_loss_fn(self):
        return None

    def attach_handlers(self):
        pass

    def run(self, train_loader, val_loader=None):
        """Run training with optional validation."""
        # Attach evaluation handler if configured
        if self.cfg.get('evaluation', {}).get('enabled', False) and self._train_evaluator is None and self._val_evaluator is None:
            self._train_evaluator, self._val_evaluator = self.create_evaluators()
            print("Created evaluators with metrics:",
                  self._train_evaluator.state.metrics if self._train_evaluator else None)
            self.attach_evaluation_handler(self.cfg.get(
                'evaluation'), train_loader, val_loader)

        trainer_cfg = self.cfg.get('trainer', {})
        run_args = trainer_cfg.get('run_args', {'max_epochs': 10})
        self.engine.run(train_loader, **run_args)

    @staticmethod
    def log_evaluation(engine, train_evaluator, val_evaluator=None, train_loader=None, val_loader=None):
        train_evaluator.run(train_loader)
        train_metrics = train_evaluator.state.metrics

        train_str = f"Training Results - Epoch: {engine.state.epoch}"
        for name, value in train_metrics.items():
            train_str += f" {name}: {value:.4f}"
        print(train_str)

        # Run val evaluator if provided
        if val_loader is not None and val_evaluator is not None:
            val_evaluator.run(val_loader)
            val_metrics = val_evaluator.state.metrics

            val_str = f"Validation Results - Epoch: {engine.state.epoch}"
            for name, value in val_metrics.items():
                val_str += f" {name}: {value:.4f}"
            print(val_str)


class PretrainTrainer(BaseTrainer):

    def get_train_step(self):
        if self.cfg.get('train_step') is not None and self.cfg.get('train_step').get('path') is not None:
            return parse_function_from_string(self.cfg.get('train_step').get('path'))
        return self.train_step

    def get_metrics(self):
        """Metrics for training engine (attached during training iterations)."""
        if self.cfg.get('metrics') is not None and self.cfg.get('metrics').get('path') is not None:
            metrics = parse_function_from_string(
                self.cfg.get('metrics').get('path'))()
            return metrics
        return None

    def get_eval_metrics(self):
        """Metrics for evaluation engines (used at epoch end)."""
        if self.cfg.get('metrics') is not None and self.cfg.get('metrics').get('path') is not None:
            metrics = parse_function_from_string(
                self.cfg.get('metrics').get('path'))()
            return metrics
        return None

    def get_loss_fn(self):
        loss_fn_cfg = self.cfg.get('loss_fn')
        if loss_fn_cfg is not None and loss_fn_cfg.get('path') is not None:
            loss_fn = parse_function_from_string(loss_fn_cfg.get('path'))()
            return loss_fn
        raise NotImplementedError("Loss function not defined in config")

    def build_optimizer(self):
        if self.cfg.get('optimizer') is not None and self.cfg.get('optimizer').get('path') is not None:
            optimizer = parse_function_from_string(
                self.cfg.get('optimizer').get('path'))
            return optimizer(self.model.parameters(), **self.cfg.get('optimizer').get('args', {}))
        return super().build_optimizer()

    def build_scheduler(self):
        if self.cfg.get('scheduler') is not None and self.cfg.get('scheduler').get('path') is not None:
            scheduler = parse_function_from_string(
                self.cfg.get('scheduler').get('path'))
            return scheduler(self.optimizer, **self.cfg.get('scheduler').get('args', {}))
        return super().build_scheduler()
