"""Defines the reinforcement learning `OnPolicyRunner`."""
import copy
import enum
import glob
import importlib.util
import inspect
import itertools
import json
import math
import os
import pathlib
import queue
import random
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set

import filelock
import numpy as np
import torch
import torch.multiprocessing as mp
from setproctitle import setproctitle as ptitle
from torch.distributions.utils import lazy_property

from allenact.algorithms.onpolicy_sync.engine import (
    TEST_MODE_STR,
    TRAIN_MODE_STR,
    VALID_MODE_STR,
    OnPolicyInference,
    OnPolicyRLEngine,
    OnPolicyTrainer,
)
from allenact.base_abstractions.callbacks import Callback
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.experiment_utils import (
    LoggingPackage,
    ScalarMeanTracker,
    set_deterministic_cudnn,
    set_seed,
)
from allenact.utils.misc_utils import (
    NumpyJSONEncoder,
    all_equal,
    get_git_diff_of_project,
)
from allenact.utils.model_utils import md5_hash_of_state_dict
from allenact.utils.system import find_free_port, get_logger
from allenact.utils.tensor_utils import SummaryWriter
from allenact.utils.viz_utils import VizSuite

CONFIG_KWARGS_STR = "__CONFIG_KWARGS__"


class SaveDirFormat(enum.Enum):
    """Directory formats that can be used when saving tensorboard logs,
    checkpoints, etc.

    during training/evaluation.
    FLAT: the first-level directories are logs, checkpoints, metrics, etc; the second-level are time strings of each experiment
    NESTED: the opposite to FLAT.
    """

    FLAT = "FLAT"
    NESTED = "NESTED"


# Has results queue (aggregated per trainer), checkpoints queue and mp context
# Instantiates train, validate, and test workers
# Logging
# Saves configs, makes folder for trainer models
class OnPolicyRunner(object):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        loaded_config_src_files: Optional[Dict[str, str]],
        seed: Optional[int] = None,
        mode: str = "train",
        deterministic_cudnn: bool = False,
        deterministic_agents: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "default",
        extra_tag: str = "",
        disable_tensorboard: bool = False,
        disable_config_saving: bool = False,
        distributed_ip_and_port: str = "127.0.0.1:0",
        distributed_preemption_threshold: float = 0.7,
        machine_id: int = 0,
        save_dir_fmt: SaveDirFormat = SaveDirFormat.FLAT,
        callbacks_paths: Optional[str] = None,
    ):
        get_logger().info(f"Runner initiated! with config: {config}")
        self.config = config
        self.output_dir = output_dir#saves checkpoints
        self.loaded_config_src_files = loaded_config_src_files
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.deterministic_cudnn = deterministic_cudnn
        self.distributed_preemption_threshold = distributed_preemption_threshold
        if multiprocessing_start_method == "default":
            if torch.cuda.is_available():
                multiprocessing_start_method = "forkserver"
            else:
                # Spawn seems to play nicer with cpus and debugging
                multiprocessing_start_method = "spawn"
        self.mp_ctx = self.init_context(mp_ctx, multiprocessing_start_method)
        self.extra_tag = extra_tag
        self.mode = mode.lower().strip()
        self.visualizer: Optional[VizSuite] = None
        self.deterministic_agents = deterministic_agents
        self.disable_tensorboard = disable_tensorboard
        self.disable_config_saving = disable_config_saving

        assert self.mode in [
            TRAIN_MODE_STR,
            TEST_MODE_STR,
        ], "Only 'train' and 'test' modes supported in runner"

        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        set_seed(self.seed)

        self.queues: Optional[Dict[str, mp.Queue]] = None

        self.processes: Dict[str, List[Union[BaseProcess, mp.Process]]] = defaultdict(
            list
        )

        self.current_checkpoint = None

        self._local_start_time_str: Optional[str] = None

        self._is_closed: bool = False

        self._collect_valid_results: bool = False

        self.distributed_ip_and_port = distributed_ip_and_port
        self.machine_id = machine_id

        self.save_dir_fmt = save_dir_fmt

        self.callbacks_paths = callbacks_paths
        get_logger().info("Everything initiated successfully.")

    @lazy_property
    def callbacks(self):
        return self.setup_callback_classes(self.callbacks_paths)

    @property
    def local_start_time_str(self) -> str:
        if self._local_start_time_str is None:
            raise RuntimeError(
                "Local start time string does not exist as neither `start_train()` or `start_test()`"
                " has been called on this runner."
            )
        return self._local_start_time_str

    @property
    def running_validation(self):
        pipeline = self.config.training_pipeline()
        return (
            sum(
                MachineParams.instance_from(
                    self.config.machine_params(VALID_MODE_STR)
                ).nprocesses
            )
            > 0
            or (
                pipeline.rollout_storage_uuid is None
                and len(pipeline.valid_pipeline_stage.loss_names) > 0
            )
        ) and self.machine_id == 0

    @staticmethod
    def init_context(
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "forkserver",
        valid_start_methods: Tuple[str, ...] = ("forkserver", "spawn", "fork"),
    ):
        if mp_ctx is None:
            get_logger().info("mp_ctx is None")
            assert multiprocessing_start_method in valid_start_methods, (
                f"multiprocessing_start_method must be one of {valid_start_methods}."
                f" Got '{multiprocessing_start_method}'"
            )

            mp_ctx = mp.get_context(multiprocessing_start_method)
        elif multiprocessing_start_method != mp_ctx.get_start_method():
            get_logger().info("mp_ctx is NOT None")
            get_logger().warning(
                f"ignoring multiprocessing_start_method '{multiprocessing_start_method}'"
                f" and using given context with '{mp_ctx.get_start_method()}'"
            )

        return mp_ctx

    def setup_callback_classes(self, callbacks: Optional[str]) -> Set[Callback]:
        """Get a list of Callback classes from a comma-separated list of files,
        paths, and/or functions.

        After separating the `callbacks` into a list of strings, each string should either
        be a:
        1. Name of a function defined on the experiment config that, when called, returns an
           object with of type `Callback`.
        2. Path to a python file containing a single class that inherits from `Callback`.
        3. Module path (e.g. `path.to.module`) where this module contains a single class that
            inherits from `Callback`.
        """
        if callbacks == "" or callbacks is None:
            return set()

        setup_dict = dict(
            name=f"{self.experiment_name}/{self.local_start_time_str}",
            config=self.config,
            mode=self.mode,
        )

        callback_objects = set()
        files = callbacks.split(",")
        for filename in files:
            # Check if the `filename` is a function on the config
            if not any(k in filename for k in [".", "/"]):
                callback_func = getattr(self.config, filename, None)
                if callback_func is not None:
                    callback = callback_func()
                    callback.setup(**setup_dict)
                    callback_objects.add(callback)
                    continue

            # Otherwise find the Callback class in the file or module
            module_path = filename.replace("/", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            module = importlib.import_module(module_path)
            classes = inspect.getmembers(module, inspect.isclass)

            callback_classes = [
                mod_class[1]
                for mod_class in classes
                if issubclass(mod_class[1], Callback)
            ]

            assert callback_classes == 1, (
                f"Expected a single callback class in {filename}, but found {len(callback_classes)}."
                f" These classes were found: {callback_classes}."
            )

            for mod_class in callback_classes:
                # NOTE: initialize the callback class
                callback = mod_class[1]()
                callback.setup(**setup_dict)
                callback_objects.add(callback)

        return callback_objects

    def _acquire_unique_local_start_time_string(self) -> str:
        """Creates a (unique) local start time string for this experiment.

        Ensures through file locks that the local start time string
        produced is unique. This implies that, if one has many
        experiments starting in parallel, at most one will be started
        every second (as the local start time string only records the
        time up to the current second).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        start_time_string_lock_path = os.path.abspath(
            os.path.join(self.output_dir, ".allenact_start_time_string.lock")
        )
        try:
            with filelock.FileLock(start_time_string_lock_path, timeout=60):
                last_start_time_string_path = os.path.join(
                    self.output_dir, ".allenact_last_start_time_string"
                )
                pathlib.Path(last_start_time_string_path).touch()

                with open(last_start_time_string_path, "r") as f:
                    last_start_time_string_list = f.readlines()

                while True:
                    candidate_str = time.strftime(
                        "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
                    )
                    if (
                        len(last_start_time_string_list) == 0
                        or last_start_time_string_list[0].strip() != candidate_str
                    ):
                        break
                    time.sleep(0.2)

                with open(last_start_time_string_path, "w") as f:
                    f.write(candidate_str)

        except filelock.Timeout as e:
            get_logger().exception(
                f"Could not acquire the lock for {start_time_string_lock_path} for 60 seconds,"
                " this suggests an unexpected deadlock. Please close all AllenAct training processes,"
                " delete this lockfile, and try again."
            )
            raise e

        assert candidate_str is not None
        return candidate_str

    def worker_devices(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode)
        )
        devices = machine_params.devices

        assert all_equal(devices) or all(
            d.index >= 0 for d in devices
        ), f"Cannot have a mix of CPU and GPU devices (`devices == {devices}`)"

        get_logger().info(f"Using {len(devices)} {mode} workers on devices {devices}")
        return devices

    def local_worker_ids(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode, machine_id=self.machine_id)
        )
        ids = machine_params.local_worker_ids

        get_logger().info(
            f"Using local worker ids {ids} (total {len(ids)} workers in machine {self.machine_id})"
        )

        return ids

    def init_visualizer(self, mode: str):
        if not self.disable_tensorboard:
            # Note: Avoid instantiating anything in machine_params (use Builder if needed)
            machine_params = MachineParams.instance_from(
                self.config.machine_params(mode)
            )
            self.visualizer = machine_params.visualizer

    @staticmethod
    def init_process(mode: str, id: int, to_close_on_termination: OnPolicyRLEngine):
        ptitle(f"{mode}-{id}")

        def create_handler(termination_type: str):
            def handler(_signo, _frame):
                prefix = f"{termination_type} signal sent to worker {mode}-{id}."
                if to_close_on_termination.is_closed:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closed, exiting."
                    )
                    sys.exit(0)
                elif not to_close_on_termination.is_closing:
                    get_logger().info(
                        f"{prefix} Forcing worker {mode}-{id} to close and exiting."
                    )
                    # noinspection PyBroadException
                    try:
                        to_close_on_termination.close(True)
                    except Exception:
                        get_logger().error(
                            f"Error occurred when closing the RL engine used by work {mode}-{id}."
                            f" We cannot recover from this and will simply exit. The exception:\n"
                            f"{traceback.format_exc()}"
                        )
                        sys.exit(1)
                    sys.exit(0)
                else:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closing, ignoring this signal."
                    )

            return handler

        signal.signal(signal.SIGTERM, create_handler("Termination"))
        signal.signal(signal.SIGINT, create_handler("Interrupt"))

    @staticmethod
    def init_worker(engine_class, args, kwargs):
        mode = kwargs["mode"]
        id = kwargs["worker_id"]

        worker = None
        try:
            worker = engine_class(*args, **kwargs)
        except Exception:
            get_logger().error(f"Encountered Exception. Terminating {mode} worker {id}")
            get_logger().exception(traceback.format_exc())
            kwargs["results_queue"].put((f"{mode}_stopped", 1 + id))
        finally:
            return worker

    @lazy_property
    def _get_callback_sensors(self) -> List[Sensor]:
        callback_sensors: List[Sensor] = []
        for c in self.callbacks:
            sensors = c.callback_sensors()
            if sensors is not None:
                callback_sensors.extend(sensors)
        return callback_sensors

    @staticmethod
    def train_loop(
        id: int = 0,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        valid_on_initial_weights: bool = False,
        *engine_args,
        **engine_kwargs,
    ):
        engine_kwargs["mode"] = TRAIN_MODE_STR
        engine_kwargs["worker_id"] = id
        engine_kwargs_for_print = {
            k: (v if k != "initial_model_state_dict" else "[SUPPRESSED]")
            for k, v in engine_kwargs.items()
        }
        get_logger().info(f"train {id} args {engine_kwargs_for_print}")

        trainer: OnPolicyTrainer = OnPolicyRunner.init_worker(
            engine_class=OnPolicyTrainer, args=engine_args, kwargs=engine_kwargs
        )
        if trainer is not None:
            OnPolicyRunner.init_process("Train", id, to_close_on_termination=trainer)
            trainer.train(
                checkpoint_file_name=checkpoint,
                restart_pipeline=restart_pipeline,
                valid_on_initial_weights=valid_on_initial_weights,
            )

    @staticmethod
    def valid_loop(id: int = 0, *engine_args, **engine_kwargs):
        get_logger().info("Here!")
        engine_kwargs["mode"] = VALID_MODE_STR
        engine_kwargs["worker_id"] = id
        get_logger().info(f"valid {id} args {engine_kwargs}")

        valid = OnPolicyRunner.init_worker(
            engine_class=OnPolicyInference, args=engine_args, kwargs=engine_kwargs
        )
        if valid is not None:
            OnPolicyRunner.init_process("Valid", id, to_close_on_termination=valid)
            valid.process_checkpoints()  # gets checkpoints via queue

    @staticmethod
    def test_loop(id: int = 0, *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = TEST_MODE_STR
        engine_kwargs["worker_id"] = id
        get_logger().info(f"test {id} args {engine_kwargs}")

        test = OnPolicyRunner.init_worker(OnPolicyInference, engine_args, engine_kwargs)
        if test is not None:
            OnPolicyRunner.init_process("Test", id, to_close_on_termination=test)
            test.process_checkpoints()  # gets checkpoints via queue

    def _initialize_start_train_or_start_test(self):
        self._is_closed = False

        if self.queues is not None:
            for k, q in self.queues.items():
                try:
                    out = q.get(timeout=1)
                    raise RuntimeError(
                        f"{k} queue was not empty before starting new training/testing (contained {out})."
                        f" This should not happen, please report how you obtained this error"
                        f" by creating an issue at https://github.com/allenai/allenact/issues."
                    )
                except queue.Empty:
                    pass

        self.queues = {
            "results": self.mp_ctx.Queue(),
            "checkpoints": self.mp_ctx.Queue(),
        }

        self._local_start_time_str = self._acquire_unique_local_start_time_string()

    def get_port(self):
        passed_port = int(self.distributed_ip_and_port.split(":")[1])
        if passed_port == 0:
            assert (
                self.machine_id == 0
            ), "Only runner with `machine_id` == 0 can search for a free port."
            distributed_port = find_free_port(
                self.distributed_ip_and_port.split(":")[0]
            )
        else:
            distributed_port = passed_port

        get_logger().info(
            f"Engines on machine_id == {self.machine_id} using port {distributed_port} and seed {self.seed}"
        )

        return distributed_port

    def start_train(
        self,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
        save_ckpt_after_every_pipeline_stage: bool = True,
        collect_valid_results: bool = False,
        valid_on_initial_weights: bool = False,
        try_restart_after_task_error: bool = False,
    ):
        get_logger().info("Starting training")
        self._initialize_start_train_or_start_test()

        self._collect_valid_results = collect_valid_results

        if not self.disable_config_saving:
            self.save_project_state()

        devices = self.worker_devices(TRAIN_MODE_STR)
        get_logger().info(f"Devices: {devices}")
        num_workers = len(devices)

        # Be extra careful to ensure that all models start
        # with the same initializations.
        set_seed(self.seed)
        get_logger().info("set_Seed executed")
        initial_model_state_dict = self.config.create_model(
            sensor_preprocessor_graph=MachineParams.instance_from(
                self.config.machine_params(self.mode)
            ).sensor_preprocessor_graph
        ).state_dict()
        get_logger().info("Model starts with same initialization")

        distributed_port = 0 if num_workers == 1 else self.get_port()

        if (
            num_workers > 1
            and "NCCL_ASYNC_ERROR_HANDLING" not in os.environ
            and "NCCL_BLOCKING_WAIT" not in os.environ
        ):
            # This ensures the NCCL distributed backend will throw errors
            # if we timeout at a call to `barrier()`
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

        worker_ids = self.local_worker_ids(TRAIN_MODE_STR)

        model_hash = None
        for trainer_id in worker_ids:
            training_kwargs = dict(
                id=trainer_id,
                checkpoint=checkpoint,
                restart_pipeline=restart_pipeline,
                experiment_name=self.experiment_name,
                config=self.config,
                callback_sensors=self._get_callback_sensors,
                results_queue=self.queues["results"],
                checkpoints_queue=self.queues["checkpoints"]
                if self.running_validation
                else None,
                checkpoints_dir=self.checkpoint_dir(),
                seed=self.seed,
                deterministic_cudnn=self.deterministic_cudnn,
                mp_ctx=self.mp_ctx,
                num_workers=num_workers,
                device=devices[trainer_id],
                distributed_ip=self.distributed_ip_and_port.split(":")[0],
                distributed_port=distributed_port,
                max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                save_ckpt_after_every_pipeline_stage=save_ckpt_after_every_pipeline_stage,
                initial_model_state_dict=initial_model_state_dict
                if model_hash is None
                else model_hash,
                first_local_worker_id=worker_ids[0],
                distributed_preemption_threshold=self.distributed_preemption_threshold,
                valid_on_initial_weights=valid_on_initial_weights,
                try_restart_after_task_error=try_restart_after_task_error,
            )
            train: BaseProcess = self.mp_ctx.Process(
                target=self.train_loop,
                kwargs=training_kwargs,
            )
            get_logger().info("Parameters assigned to process")
            try:
                get_logger().info("train.start()")
                train.start()
            except (ValueError, OSError, ConnectionRefusedError, EOFError) as e:
                # If the `initial_model_state_dict` is too large we sometimes
                # run into errors passing it with multiprocessing. In such cases
                # we instead hash the state_dict and confirm, in each engine worker, that
                # this hash equals the model the engine worker instantiates.
                if (
                    (isinstance(e, ValueError) and e.args[0] == "too many fds")
                    or (isinstance(e, OSError) and e.errno == 22)
                    or (isinstance(e, ConnectionRefusedError) and e.errno == 111)
                    or isinstance(e, EOFError)
                ):
                    model_hash = md5_hash_of_state_dict(initial_model_state_dict)
                    training_kwargs["initial_model_state_dict"] = model_hash
                    train = self.mp_ctx.Process(
                        target=self.train_loop,
                        kwargs=training_kwargs,
                    )
                    train.start()
                else:
                    raise e

            self.processes[TRAIN_MODE_STR].append(train)

        get_logger().info(
            f"Started {len(self.processes[TRAIN_MODE_STR])} train processes"
        )

        # Validation
        if self.running_validation:
            device = self.worker_devices(VALID_MODE_STR)[0]
            self.init_visualizer(VALID_MODE_STR)
            valid: BaseProcess = self.mp_ctx.Process(
                target=self.valid_loop,
                args=(0,),
                kwargs=dict(
                    config=self.config,
                    callback_sensors=self._get_callback_sensors,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
                    deterministic_agents=self.deterministic_agents,
                    mp_ctx=self.mp_ctx,
                    device=device,
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                ),
            )
            valid.start()
            self.processes[VALID_MODE_STR].append(valid)

            get_logger().info(
                f"Started {len(self.processes[VALID_MODE_STR])} valid processes"
            )
        else:
            get_logger().info(
                "No processes allocated to validation, no validation will be run."
            )

        metrics_file_template: Optional[str] = None

        if self._collect_valid_results:
            metrics_dir = self.metric_path(self.local_start_time_str)
            os.makedirs(metrics_dir, exist_ok=True)
            suffix = f"__valid_{self.local_start_time_str}"
            metrics_file_template = os.path.join(
                metrics_dir, "metrics" + suffix + "{:012d}.json"
            )  # template for training steps

            get_logger().info(
                f"Saving valid metrics with template {metrics_file_template}"
            )

            # Check output file can be written
            with open(metrics_file_template.format(0), "w") as f:
                json.dump([], f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        valid_results = self.log_and_close(
            start_time_str=self.local_start_time_str,
            nworkers=len(worker_ids),  # TODO num_workers once we forward metrics,
            metrics_file=metrics_file_template,
        )

        if not self._collect_valid_results:
            return self.local_start_time_str
        else:
            return self.local_start_time_str, valid_results

    def start_test(
        self,
        checkpoint_path_dir_or_pattern: str,
        infer_output_dir: bool = False,
        approx_ckpt_step_interval: Optional[Union[float, int]] = None,
        max_sampler_processes_per_worker: Optional[int] = None,
        inference_expert: bool = False,
    ) -> List[Dict]:
        # Tester always runs on a single machine
        assert (
            self.machine_id == 0
        ), f"Received `machine_id={self.machine_id} for test. Only one machine supported."
        assert isinstance(
            checkpoint_path_dir_or_pattern, str
        ), "Must provide a --checkpoint path or pattern to test on."

        self.extra_tag += (
            "__" * (len(self.extra_tag) > 0) + "enforced_test_expert"
        ) * inference_expert
        self._initialize_start_train_or_start_test()

        devices = self.worker_devices(TEST_MODE_STR)
        self.init_visualizer(TEST_MODE_STR)
        num_testers = len(devices)

        distributed_port = 0
        if num_testers > 1:
            distributed_port = find_free_port()

        # Tester always runs on a single machine
        for tester_it in range(num_testers):
            test: BaseProcess = self.mp_ctx.Process(
                target=self.test_loop,
                args=(tester_it,),
                kwargs=dict(
                    config=self.config,
                    callback_sensors=self._get_callback_sensors,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
                    deterministic_agents=self.deterministic_agents,
                    mp_ctx=self.mp_ctx,
                    num_workers=num_testers,
                    device=devices[tester_it],
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                    distributed_port=distributed_port,
                    enforce_expert=inference_expert,
                ),
            )

            test.start()
            self.processes[TEST_MODE_STR].append(test)

        get_logger().info(
            f"Started {len(self.processes[TEST_MODE_STR])} test processes"
        )

        checkpoint_paths = self.get_checkpoint_files(
            checkpoint_path_dir_or_pattern=checkpoint_path_dir_or_pattern,
            approx_ckpt_step_interval=approx_ckpt_step_interval,
        )
        steps = [self.step_from_checkpoint(cp) for cp in checkpoint_paths]

        get_logger().info(f"Running test on {len(steps)} steps {steps}")

        for checkpoint_path in checkpoint_paths:
            # Make all testers work on each checkpoint
            for tester_it in range(num_testers):
                self.queues["checkpoints"].put(("eval", checkpoint_path))

        # Signal all testers to terminate cleanly
        for _ in range(num_testers):
            self.queues["checkpoints"].put(("quit", None))

        if self.save_dir_fmt == SaveDirFormat.NESTED:
            if infer_output_dir:  # NOTE: we change output_dir here
                self.output_dir = self.checkpoint_log_folder_str(checkpoint_paths[0])
            suffix = ""
        elif self.save_dir_fmt == SaveDirFormat.FLAT:
            suffix = f"__test_{self.local_start_time_str}"
        else:
            raise NotImplementedError
        metrics_dir = self.metric_path(self.local_start_time_str)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_file_path = os.path.join(metrics_dir, "metrics" + suffix + ".json")

        get_logger().info(f"Saving test metrics in {metrics_file_path}")

        # Check output file can be written
        with open(metrics_file_path, "w") as f:
            json.dump([], f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        return self.log_and_close(
            start_time_str=self.checkpoint_start_time_str(checkpoint_paths[0]),
            nworkers=num_testers,
            test_steps=steps,
            metrics_file=metrics_file_path,
        )

    @staticmethod
    def checkpoint_start_time_str(checkpoint_file_name):
        parts = checkpoint_file_name.split(os.path.sep)
        assert len(parts) > 1, f"{checkpoint_file_name} is not a valid checkpoint path"
        start_time_str = parts[-2]
        get_logger().info(f"Using checkpoint start time {start_time_str}")
        return start_time_str

    @staticmethod
    def checkpoint_log_folder_str(checkpoint_file_name):
        parts = checkpoint_file_name.split(os.path.sep)
        assert len(parts) > 1, f"{checkpoint_file_name} is not a valid checkpoint path"
        log_folder_str = os.path.sep.join(parts[:-2])  # remove checkpoints/*.pt
        get_logger().info(f"Using log folder {log_folder_str}")
        return log_folder_str

    @property
    def experiment_name(self):
        if len(self.extra_tag) > 0:
            return f"{self.config.tag()}_{self.extra_tag}"
        return self.config.tag()

    #responsible for saving chackpoints in a particular place
    def checkpoint_dir(
        self, start_time_str: Optional[str] = None, create_if_none: bool = True
    ):
        path_parts = [
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            start_time_str or self.local_start_time_str,
        ]
        if self.save_dir_fmt == SaveDirFormat.NESTED:
            folder = os.path.join(
                self.output_dir,
                *path_parts,
                "checkpoints",
            )
        elif self.save_dir_fmt == SaveDirFormat.FLAT:
            folder = os.path.join(
                self.output_dir,
                "checkpoints",
                *path_parts,
            )
        else:
            raise NotImplementedError
        if create_if_none:
            os.makedirs(folder, exist_ok=True)
        return folder

    def log_writer_path(self, start_time_str: str) -> str:
        if self.save_dir_fmt == SaveDirFormat.NESTED:
            if self.mode == TEST_MODE_STR:
                return os.path.join(
                    self.output_dir,
                    "test",
                    self.config.tag(),
                    self.local_start_time_str,
                )
            path = os.path.join(
                self.output_dir,
                self.config.tag()
                if self.extra_tag == ""
                else os.path.join(self.config.tag(), self.extra_tag),
                start_time_str,
                "train_tb",
            )
            return path
        elif self.save_dir_fmt == SaveDirFormat.FLAT:
            path = os.path.join(
                self.output_dir,
                "tb",
                self.config.tag()
                if self.extra_tag == ""
                else os.path.join(self.config.tag(), self.extra_tag),
                start_time_str,
            )
            if self.mode == TEST_MODE_STR:
                path = os.path.join(path, "test", self.local_start_time_str)
            return path
        else:
            raise NotImplementedError

    def metric_path(self, start_time_str: str) -> str:
        if self.save_dir_fmt == SaveDirFormat.NESTED:
            return os.path.join(
                self.output_dir,
                "test",
                self.config.tag(),
                start_time_str,
            )
        elif self.save_dir_fmt == SaveDirFormat.FLAT:
            return os.path.join(
                self.output_dir,
                "metrics",
                self.config.tag()
                if self.extra_tag == ""
                else os.path.join(self.config.tag(), self.extra_tag),
                start_time_str,
            )
        else:
            raise NotImplementedError

    def save_project_state(self):
        path_parts = [
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            self.local_start_time_str,
        ]
        if self.save_dir_fmt == SaveDirFormat.NESTED:
            base_dir = os.path.join(
                self.output_dir,
                *path_parts,
                "used_configs",
            )
        elif self.save_dir_fmt == SaveDirFormat.FLAT:
            base_dir = os.path.join(
                self.output_dir,
                "used_configs",
                *path_parts,
            )
        else:
            raise NotImplementedError
        os.makedirs(base_dir, exist_ok=True)

        # Saving current git diff
        try:
            sha, diff_str = get_git_diff_of_project()
            with open(os.path.join(base_dir, f"{sha}.patch"), "w") as f:
                f.write(diff_str)

            get_logger().info(f"Git diff saved to {base_dir}")
        except subprocess.CalledProcessError:
            get_logger().warning(
                "Failed to get a git diff of the current project."
                f" Is it possible that {os.getcwd()} is not under version control?"
            )

        # Saving configs
        if self.loaded_config_src_files is not None:
            for src_path in self.loaded_config_src_files:
                if src_path == CONFIG_KWARGS_STR:
                    # We also save key-word arguments passed to the experiment
                    # initializer.
                    save_path = os.path.join(base_dir, "config_kwargs.json")
                    assert not os.path.exists(
                        save_path
                    ), f"{save_path} should not already exist."
                    with open(save_path, "w") as f:
                        json.dump(json.loads(self.loaded_config_src_files[src_path]), f)
                    continue

                assert os.path.isfile(src_path), f"Config file {src_path} not found"
                src_path = os.path.abspath(src_path)

                # To prevent overwriting files with the same name, we loop
                # here until we find a prefix (if necessary) to prevent
                # name collisions.
                k = -1
                while True:
                    prefix = "" if k == -1 else f"namecollision{k}__"
                    k += 1
                    dst_path = os.path.join(
                        base_dir,
                        f"{prefix}{os.path.basename(src_path)}",
                    )
                    if not os.path.exists(dst_path):
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        with open(src_path, "r") as f:
                            file_contents = f.read()
                        with open(dst_path, "w") as f:
                            f.write(
                                f"### THIS FILE ORIGINALLY LOCATED AT '{src_path}'\n\n{file_contents}"
                            )
                        break

        get_logger().info(f"Config files saved to {base_dir}")
        for callback in self.callbacks:
            callback.after_save_project_state(base_dir=base_dir)

    def _update_keys(
        self,
        d: Union[Dict[str, Any], str],
        tag_if_not_a_loss: str,
        mode: str,
        stage_component_uuid: Optional[str] = None,
    ) -> Union[Dict[str, Any], str]:
        midfix = "-" if stage_component_uuid is None else f"-{stage_component_uuid}-"

        def _convert(key: str):
            if key.startswith("losses/"):
                return f"{mode}{midfix}{key}"
            else:
                return f"{mode}{midfix}{tag_if_not_a_loss}/{key}"

        if isinstance(d, str):
            return _convert(d)
        return {_convert(k): v for k, v in d.items()}

    def _process_logging_packages(
        self,
        log_writer: Optional[SummaryWriter],
        pkgs: Union[LoggingPackage, List[LoggingPackage]],
        last_steps: Optional[int],
        last_storage_uuid_to_total_experiences: Optional[Dict[str, int]],
        last_time: Optional[float],
        all_results: Optional[List[Any]] = None,
    ):
        mode = pkgs[0].mode
        assert all(
            pkg.mode == mode for pkg in pkgs
        ), "All logging packages must be the same mode."
        assert mode == self.mode or (
            mode == VALID_MODE_STR and self.mode == TRAIN_MODE_STR
        ), (
            "Logging package mode must match the logger mode except when training where the logging package may"
            "be of mode 'valid'."
        )
        training = mode == TRAIN_MODE_STR  # Are we logging training packages

        current_time = time.time()

        training_steps = pkgs[0].training_steps
        storage_uuid_to_total_experiences = pkgs[0].storage_uuid_to_total_experiences
        callback_metric_means = dict()

        def update_keys_misc(
            key_or_dict: Union[str, Dict[str, Any]],
            stage_component_uuid: Optional[str] = None,
        ):
            # Important to use mode and not self.mode here
            return self._update_keys(
                d=key_or_dict,
                tag_if_not_a_loss="misc",
                mode=mode,
                stage_component_uuid=stage_component_uuid,
            )

        def update_keys_metric(
            key_or_dict: Union[str, Dict[str, Any]],
            stage_component_uuid: Optional[str] = None,
        ):
            # Important to use mode and not self.mode here
            return self._update_keys(
                d=key_or_dict,
                tag_if_not_a_loss="metrics",
                mode=mode,
                stage_component_uuid=stage_component_uuid,
            )

        if training and log_writer is not None:
            log_writer.add_scalar(
                tag=update_keys_misc("pipeline_stage"),
                scalar_value=pkgs[0].pipeline_stage,
                global_step=training_steps,
            )
        callback_metric_means[update_keys_misc("pipeline_stage")] = pkgs[
            0
        ].pipeline_stage

        storage_uuid_to_total_experiences_key = {}
        for storage_uuid, val in storage_uuid_to_total_experiences.items():
            total_experiences_key = update_keys_misc(
                f"{storage_uuid}_total_experiences"
            )
            storage_uuid_to_total_experiences_key[storage_uuid] = total_experiences_key

            if training and log_writer is not None:
                log_writer.add_scalar(
                    tag=total_experiences_key,
                    scalar_value=val,
                    global_step=training_steps,
                )
            callback_metric_means[total_experiences_key] = val

        metrics_and_info_tracker = ScalarMeanTracker()
        scalar_name_to_total_storage_experience = {}
        scalar_name_to_total_experiences_key = {}
        storage_uuid_to_stage_component_uuids = defaultdict(lambda: set())
        metric_dicts_list, render, checkpoint_file_name = [], {}, []
        tasks_callback_data = []

        for pkg in pkgs:
            metrics_and_info_tracker.add_scalars(
                scalars=update_keys_metric(pkg.metrics_tracker.means()),
                n=update_keys_metric(pkg.metrics_tracker.counts()),
            )
            tasks_callback_data.extend(pkg.task_callback_data)
            metric_dicts_list.extend(pkg.metric_dicts)
            if pkg.viz_data is not None:
                render.update(pkg.viz_data)
            checkpoint_file_name.append(pkg.checkpoint_file_name)

            for (
                (stage_component_uuid, storage_uuid),
                info_tracker,
            ) in pkg.info_trackers.items():

                if stage_component_uuid is not None:
                    storage_uuid_to_stage_component_uuids[storage_uuid].add(
                        stage_component_uuid
                    )

                info_means = update_keys_misc(
                    info_tracker.means(),
                    stage_component_uuid,
                )
                info_counts = update_keys_misc(
                    info_tracker.counts(),
                    stage_component_uuid,
                )
                metrics_and_info_tracker.add_scalars(
                    scalars=info_means,
                    n=info_counts,
                )

                total_exp_for_storage = pkg.storage_uuid_to_total_experiences[
                    storage_uuid
                ]

                if stage_component_uuid is None:
                    assert total_exp_for_storage == training_steps

                for scalar_name in info_means:
                    if scalar_name in scalar_name_to_total_storage_experience:
                        assert (
                            total_exp_for_storage
                            == scalar_name_to_total_storage_experience[scalar_name]
                        ), (
                            f"For metric {scalar_name}: there is disagreement between the training steps parameter"
                            f" across different workers ({total_exp_for_storage} !="
                            f" {scalar_name_to_total_storage_experience[scalar_name]}). This suggests an error in "
                            f" AllenAct, please report this issue at https://github.com/allenai/allenact/issues."
                        )
                    else:
                        scalar_name_to_total_storage_experience[
                            scalar_name
                        ] = total_exp_for_storage
                        scalar_name_to_total_experiences_key[
                            scalar_name
                        ] = storage_uuid_to_total_experiences_key[storage_uuid]

        assert all_equal(
            checkpoint_file_name
        ), f"All {mode} logging packages must have the same checkpoint_file_name."

        message = [
            f"{mode.upper()}: {training_steps} rollout steps ({pkgs[0].storage_uuid_to_total_experiences})"
        ]
        metrics_and_info_means = metrics_and_info_tracker.means()
        callback_metric_means.update(metrics_and_info_means)

        for k in sorted(
            metrics_and_info_means.keys(),
            key=lambda mean_key: (mean_key.count("/"), mean_key),
        ):
            if log_writer is not None:
                log_writer.add_scalar(
                    tag=k,
                    scalar_value=metrics_and_info_means[k],
                    global_step=scalar_name_to_total_storage_experience.get(
                        k, training_steps
                    ),
                )
            short_key = (
                "/".join(k.split("/")[1:])
                if k.startswith(f"{mode}-") and "/" in k
                else k
            )
            message.append(f"{short_key} {metrics_and_info_means[k]:.3g}")

        if training:
            # Log information about FPS and EPS (experiences per second, for non-rollout storage).
            # Not needed during testing or validation.
            message += [f"elapsed_time {(current_time - last_time):.3g}s"]

            if last_steps > 0:
                fps = (training_steps - last_steps) / (current_time - last_time)
                message += [f"approx_fps {fps:.3g}"]
                approx_fps_key = update_keys_misc("approx_fps")
                if log_writer is not None:
                    log_writer.add_scalar(approx_fps_key, fps, training_steps)
                callback_metric_means[approx_fps_key] = fps

            for (
                storage_uuid,
                last_total_exp,
            ) in last_storage_uuid_to_total_experiences.items():
                if storage_uuid in storage_uuid_to_total_experiences:
                    cur_total_exp = storage_uuid_to_total_experiences[storage_uuid]
                    eps = (cur_total_exp - last_total_exp) / (current_time - last_time)
                    message += [f"{storage_uuid}/approx_eps {eps:.3g}"]
                    for stage_component_uuid in storage_uuid_to_stage_component_uuids[
                        storage_uuid
                    ]:
                        approx_eps_key = update_keys_misc(
                            f"approx_eps",
                            stage_component_uuid,
                        )
                        callback_metric_means[approx_eps_key] = eps
                        scalar_name_to_total_experiences_key[
                            approx_eps_key
                        ] = storage_uuid_to_total_experiences_key[storage_uuid]

                        if log_writer is not None:
                            log_writer.add_scalar(
                                approx_eps_key,
                                eps,
                                cur_total_exp,
                            )

        metrics_and_info_means_with_metrics_dicts_list = copy.deepcopy(
            metrics_and_info_means
        )
        metrics_and_info_means_with_metrics_dicts_list.update(
            {"training_steps": training_steps, "tasks": metric_dicts_list}
        )
        if all_results is not None:
            all_results.append(metrics_and_info_means_with_metrics_dicts_list)

        num_tasks = sum([pkg.num_non_empty_metrics_dicts_added for pkg in pkgs])
        num_tasks_completed_key = update_keys_misc("num_tasks_completed_since_last_log")
        if log_writer is not None:
            log_writer.add_scalar(num_tasks_completed_key, num_tasks, training_steps)
        callback_metric_means[num_tasks_completed_key] = num_tasks

        message.append(f"new_tasks_completed {num_tasks}")
        if not training:
            message.append(f"checkpoint {checkpoint_file_name[0]}")

        get_logger().info(" ".join(message))

        for callback in self.callbacks:
            if mode == TRAIN_MODE_STR:
                callback.on_train_log(
                    metrics=metric_dicts_list,
                    metric_means=callback_metric_means,
                    step=training_steps,
                    tasks_data=tasks_callback_data,
                    scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
                )

            if mode == VALID_MODE_STR:
                callback.on_valid_log(
                    metrics=metrics_and_info_means_with_metrics_dicts_list,
                    metric_means=callback_metric_means,
                    step=training_steps,
                    checkpoint_file_name=checkpoint_file_name[0],
                    tasks_data=tasks_callback_data,
                    scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
                )

            if mode == TEST_MODE_STR:
                callback.on_test_log(
                    metrics=metrics_and_info_means_with_metrics_dicts_list,
                    metric_means=callback_metric_means,
                    step=training_steps,
                    checkpoint_file_name=checkpoint_file_name[0],
                    tasks_data=tasks_callback_data,
                    scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
                )

        if self.visualizer is not None:
            self.visualizer.log(
                log_writer=log_writer,
                task_outputs=metric_dicts_list,
                render=render,
                num_steps=training_steps,
            )

        return training_steps, storage_uuid_to_total_experiences, current_time

    def process_valid_package(
        self,
        log_writer: Optional[SummaryWriter],
        pkg: LoggingPackage,
        all_results: Optional[List[Any]] = None,
    ):
        return self._process_logging_packages(
            log_writer=log_writer,
            pkgs=[pkg],
            last_steps=None,
            last_storage_uuid_to_total_experiences=None,
            last_time=None,
            all_results=all_results,
        )

    def process_train_packages(
        self,
        log_writer: Optional[SummaryWriter],
        pkgs: List[LoggingPackage],
        last_steps: int,
        last_storage_uuid_to_total_experiences: Dict[str, int],
        last_time: float,
    ):
        return self._process_logging_packages(
            log_writer=log_writer,
            pkgs=pkgs,
            last_steps=last_steps,
            last_storage_uuid_to_total_experiences=last_storage_uuid_to_total_experiences,
            last_time=last_time,
        )

    def process_test_packages(
        self,
        log_writer: Optional[SummaryWriter],
        pkgs: List[LoggingPackage],
        all_results: Optional[List[Any]] = None,
    ):
        return self._process_logging_packages(
            log_writer=log_writer,
            pkgs=pkgs,
            last_steps=None,
            last_storage_uuid_to_total_experiences=None,
            last_time=None,
            all_results=all_results,
        )

    def log_and_close(
        self,
        start_time_str: str,
        nworkers: int,
        test_steps: Sequence[int] = (),
        metrics_file: Optional[str] = None,
    ) -> List[Dict]:
        ptitle(f"AllenAct-Logging-{self.local_start_time_str}")
        finalized = False

        log_writer: Optional[SummaryWriter] = None
        if not self.disable_tensorboard:
            log_writer = SummaryWriter(
                log_dir=self.log_writer_path(start_time_str),
                filename_suffix=f"__{self.mode}_{self.local_start_time_str}",
            )

        # To aggregate/buffer metrics from trainers/testers
        collected: List[LoggingPackage] = []
        last_train_steps = 0
        last_storage_uuid_to_total_experiences = {}
        last_train_time = time.time()
        # test_steps = sorted(test_steps, reverse=True)
        eval_results: List[Dict] = []
        unfinished_workers = nworkers

        try:
            while True:
                try:
                    package: Union[
                        LoggingPackage, Union[Tuple[str, Any], Tuple[str, Any, Any]]
                    ] = self.queues["results"].get(timeout=1)

                    if isinstance(package, LoggingPackage):
                        pkg_mode = package.mode

                        if pkg_mode == TRAIN_MODE_STR:
                            collected.append(package)
                            if len(collected) >= nworkers:

                                collected = sorted(
                                    collected,
                                    key=lambda pkg: (
                                        pkg.training_steps,
                                        *sorted(
                                            pkg.storage_uuid_to_total_experiences.items()
                                        ),
                                    ),
                                )

                                if (
                                    collected[nworkers - 1].training_steps
                                    == collected[0].training_steps
                                    and collected[
                                        nworkers - 1
                                    ].storage_uuid_to_total_experiences
                                    == collected[0].storage_uuid_to_total_experiences
                                ):  # ensure all workers have provided the same training_steps and total_experiences
                                    (
                                        last_train_steps,
                                        last_storage_uuid_to_total_experiences,
                                        last_train_time,
                                    ) = self.process_train_packages(
                                        log_writer=log_writer,
                                        pkgs=collected[:nworkers],
                                        last_steps=last_train_steps,
                                        last_storage_uuid_to_total_experiences=last_storage_uuid_to_total_experiences,
                                        last_time=last_train_time,
                                    )
                                    collected = collected[nworkers:]
                                elif len(collected) > 2 * nworkers:
                                    get_logger().warning(
                                        f"Unable to aggregate train packages from all {nworkers} workers"
                                        f"after {len(collected)} packages collected"
                                    )
                        elif (
                            pkg_mode == VALID_MODE_STR
                        ):  # they all come from a single worker
                            if (
                                package.training_steps is not None
                            ):  # no validation samplers
                                self.process_valid_package(
                                    log_writer=log_writer,
                                    pkg=package,
                                    all_results=eval_results
                                    if self._collect_valid_results
                                    else None,
                                )

                                if metrics_file is not None:
                                    with open(
                                        metrics_file.format(package.training_steps), "w"
                                    ) as f:
                                        json.dump(
                                            eval_results[-1],
                                            f,
                                            indent=4,
                                            sort_keys=True,
                                            cls=NumpyJSONEncoder,
                                        )
                                        get_logger().info(
                                            "Written valid results file {}".format(
                                                metrics_file.format(
                                                    package.training_steps
                                                ),
                                            )
                                        )

                            if (
                                finalized and self.queues["checkpoints"].empty()
                            ):  # assume queue is actually empty after trainer finished and no checkpoints in queue
                                break
                        elif pkg_mode == TEST_MODE_STR:
                            collected.append(package)
                            if len(collected) >= nworkers:
                                collected = sorted(
                                    collected, key=lambda x: x.training_steps
                                )  # sort by num_steps
                                if (
                                    collected[nworkers - 1].training_steps
                                    == collected[0].training_steps
                                ):  # ensure nworkers have provided the same num_steps
                                    self.process_test_packages(
                                        log_writer=log_writer,
                                        pkgs=collected[:nworkers],
                                        all_results=eval_results,
                                    )

                                    collected = collected[nworkers:]
                                    with open(metrics_file, "w") as f:
                                        json.dump(
                                            eval_results,
                                            f,
                                            indent=4,
                                            sort_keys=True,
                                            cls=NumpyJSONEncoder,
                                        )
                                        get_logger().info(
                                            f"Updated {metrics_file} up to checkpoint"
                                            f" {test_steps[len(eval_results) - 1]}"
                                        )
                        else:
                            get_logger().error(
                                f"Runner received unknown package of type {pkg_mode}"
                            )
                    else:
                        pkg_mode = package[0]

                        if pkg_mode == "train_stopped":
                            if package[1] == 0:
                                finalized = True
                                if not self.running_validation:
                                    get_logger().info(
                                        "Terminating runner after trainer done (no validation)"
                                    )
                                    break
                            else:
                                raise Exception(
                                    f"Train worker {package[1] - 1} abnormally terminated"
                                )
                        elif pkg_mode == "valid_stopped":
                            raise Exception(
                                f"Valid worker {package[1] - 1} abnormally terminated"
                            )
                        elif pkg_mode == "test_stopped":
                            if package[1] == 0:
                                unfinished_workers -= 1
                                if unfinished_workers == 0:
                                    get_logger().info(
                                        "Last tester finished. Terminating"
                                    )
                                    finalized = True
                                    break
                            else:
                                raise RuntimeError(
                                    f"Test worker {package[1] - 1} abnormally terminated"
                                )
                        else:
                            get_logger().error(
                                f"Runner received invalid package tuple {package}"
                            )
                except queue.Empty as _:
                    if all(
                        p.exitcode is not None
                        for p in itertools.chain(*self.processes.values())
                    ):
                        break
        except KeyboardInterrupt:
            get_logger().info("KeyboardInterrupt. Terminating runner.")
        except Exception:
            get_logger().error("Encountered Exception. Terminating runner.")
            get_logger().exception(traceback.format_exc())
        finally:
            if finalized:
                get_logger().info("Done")
            if log_writer is not None:
                log_writer.close()
            self.close()
            return eval_results

    def get_checkpoint_files(
        self,
        checkpoint_path_dir_or_pattern: str,
        approx_ckpt_step_interval: Optional[int] = None,
    ):

        if os.path.isdir(checkpoint_path_dir_or_pattern):
            # The fragment is a path to a directory, lets use this directory
            # as the base dir to search for checkpoints
            checkpoint_path_dir_or_pattern = os.path.join(
                checkpoint_path_dir_or_pattern, "*.pt"
            )

        ckpt_paths = glob.glob(checkpoint_path_dir_or_pattern, recursive=True)

        if len(ckpt_paths) == 0:
            raise FileNotFoundError(
                f"Could not find any checkpoints at {os.path.abspath(checkpoint_path_dir_or_pattern)}, is it possible"
                f" the path has been mispecified?"
            )

        step_count_ckpt_pairs = [(self.step_from_checkpoint(p), p) for p in ckpt_paths]
        step_count_ckpt_pairs.sort()
        ckpts_paths = [p for _, p in step_count_ckpt_pairs]
        step_counts = np.array([sc for sc, _ in step_count_ckpt_pairs])

        if approx_ckpt_step_interval is not None:
            assert (
                approx_ckpt_step_interval > 0
            ), "`approx_ckpt_step_interval` must be >0"
            inds_to_eval = set()
            for i in range(
                math.ceil(step_count_ckpt_pairs[-1][0] / approx_ckpt_step_interval) + 1
            ):
                inds_to_eval.add(
                    int(np.argmin(np.abs(step_counts - i * approx_ckpt_step_interval)))
                )

            ckpts_paths = [ckpts_paths[ind] for ind in sorted(list(inds_to_eval))]
        return ckpts_paths

    @staticmethod
    def step_from_checkpoint(ckpt_path: str) -> int:
        parts = os.path.basename(ckpt_path).split("__")
        for part in parts:
            if "steps_" in part:
                possible_num = part.split("_")[-1].split(".")[0]
                if possible_num.isdigit():
                    return int(possible_num)

        get_logger().warning(
            f"The checkpoint {os.path.basename(ckpt_path)} does not follow the checkpoint naming convention"
            f" used by AllenAct. As a fall back we must load the checkpoint into memory to find the"
            f" training step count, this may increase startup time if the checkpoints are large or many"
            f" must be loaded in sequence."
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt["total_steps"]

    def close(self, verbose=True):
        if self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    get_logger().info(s)
                elif isinstance(s, Exception):
                    get_logger().exception(traceback.format_exc())
                else:
                    raise NotImplementedError()

        # First send termination signals
        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                if process.is_alive():
                    logif(f"Terminating {process_type} {it}")
                    process.terminate()

        # Now join processes
        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                try:
                    logif(f"Joining {process_type} {it}")
                    process.join(1)
                    logif(f"Closed {process_type} {it}")
                except Exception as e:
                    logif(f"Exception raised when closing {process_type} {it}")
                    logif(e)

        self.processes.clear()
        self._is_closed = True

    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)
