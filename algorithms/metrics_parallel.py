import csv
import os
import queue
import threading

import numpy as np

COLUMNS = [
    "environment_steps",
    "updates",
    "episode_returns_mean",
    "entropy_mean",
    "loss_mean",
    "actor_loss_mean",
    "value_loss_mean",
    "reward_std_mean",
# behavioral signals (commentary)
    "deliveries_mean",
    "block_rate_mean",
    "idle_rate_mean",
    "pickup_rate_mean",
    "deliveries_early_mean",
    "deliveries_mid_mean",
    "deliveries_late_mean",
    "block_early_mean",
    "block_mid_mean",
    "block_late_mean",
]

COLUMNS_DQN = [
    "environment_steps",
    "updates",
    "episode_returns_mean",
    "loss_mean",
    "epsilon",
    "reward_std_mean",
    "deliveries_mean",
    "block_rate_mean",
    "idle_rate_mean",
    "pickup_rate_mean",
    "deliveries_early_mean",
    "deliveries_mid_mean",
    "deliveries_late_mean",
    "block_early_mean",
    "block_mid_mean",
    "block_late_mean",
]

class AsyncCSVLogger:
    def __init__(self, path, dict_keys, full_metrics=False, resume=False,isdqn=False):
        self.full_metrics = full_metrics
        self.dict_keys = dict_keys
        self.isdqn = isdqn

        # --- File setup ---

        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        write_header = not (
            resume and os.path.isfile(path) and os.path.getsize(path) > 0
        )

        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)

        if write_header:
            if isdqn:
                self._w.writerow(dict_keys if full_metrics else COLUMNS_DQN)
            else:
                self._w.writerow(dict_keys if full_metrics else COLUMNS)
            self._f.flush()
        # --- Async logging ---

        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            raw = self._queue.get()
            if raw is None:
                break
            self._write_row(raw)

    def _write_row(self, raw):
        if self.full_metrics:
            row = [raw.get(c, f"{np.finfo(np.float32).max}") for c in self.dict_keys]
        else:
            if self.isdqn:
                row = [raw.get(c, "") for c in COLUMNS]
            else:
                row = [raw.get(c, "") for c in COLUMNS_DQN]
        self._w.writerow(row)
        self._f.flush()

    def log(self, raw: dict):
        self._queue.put(raw)

    def close(self):
        self._queue.put(None)
        self._thread.join()
        try:
            self._f.close()
        except Exception:
            pass


from concurrent.futures import Future, ProcessPoolExecutor
from algorithms.metrics_handler import process_raw as process_block
from algorithms.metrics_funcs import create_dummy_metrics, sort_metrics

from concurrent.futures import ThreadPoolExecutor



def init_Asylogger(
    path: str, E: int, N: int, A: int, full_metrics: bool = False, resume: bool = False,isdqn:bool=False
):

    def init_l(E, N, A):
        if isdqn:
            dummy_metric = create_dummy_metrics(E, N, A)
            init_columns = process_block(dummy_metric, 1, 1, actionDim=A)
        else:
            dummy_metric = create_dummy_metrics(E, N, A)
            init_columns = process_block(dummy_metric, 1, 1, actionDim=A)

        key_list = [str(k) for k, _ in init_columns[0].items()]
        sorted_key_list = sort_metrics(key_list)
        logger = AsyncCSVLogger(path, sorted_key_list, full_metrics, resume,isdqn)
        return logger
    
    executor = ThreadPoolExecutor(max_workers=1)
    logger_future = executor.submit(init_l,E, N, A)
 
    
    return logger_future

import queue
import threading
from concurrent.futures import ProcessPoolExecutor

class BlockProcessor:
    def __init__(self, logger, max_workers=4):
        self.logger = logger
        self.tasks = queue.Queue()
        self.stop_flag = False

        self.pool = ProcessPoolExecutor(max_workers=max_workers)

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def add_task(self, args_task):
        self.tasks.put(args_task)

    def close(self):
        self.stop_flag = True
        self.tasks.put(None)   # wake the worker
        self.thread.join()
        self.pool.shutdown(wait=True)

    def _worker(self):
        while True:
            args = self.tasks.get()

            if args is None and self.stop_flag:
                break

            if args is None:
                continue

            # submit to process pool
            future = self.pool.submit(process_block, *args)
            block = future.result()  # list[dict]

            for row in block:
                if isinstance(self.logger,Future):
                    logger=self.logger.result()
                    logger.log(row)
                else:
                    self.logger.log(row)
