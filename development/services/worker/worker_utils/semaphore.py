import os
import threading
import time
from loguru import logger

from wredis.hash import RedisHashManager
import yaml


class SharedResource:
    def __init__(self, hash_manager: RedisHashManager = None):
        # Initialized to 1, acts like a lock
        NUM_CURRENT_TRAIN = max(int(os.environ.get("NUM_CURRENT_TRAIN", 1)), 1)

        self.semaphore = threading.Semaphore(NUM_CURRENT_TRAIN)
        self.resource_available = True
        self.end_time = 0
        self.start_time = 0

        self.hash_manager = hash_manager

    def elapsedtime(self):
        return self.end_time - self.start_time

    def notify_training_has_started(self, args_dict: dict):
        """
        Notify the number of current training jobs.

        Args:
            args_dict: dict: Dictionary containing the arguments for the training job.
        """

        try:
            this_worker_ip = os.environ.get("WORKER_HOST")
            
            config_path = args_dict["config_path"]
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)  # Convertir YAML a dict

            experiment_name = user_config.get("sweeper", {}).get("study_name", None)
            user_code = args_dict.get("user_code", None)
            task_id = args_dict.get("task_id", None)
            ttl_timeout = 60 * 60 * 24 * 30  # 30 days

            # año-mes-dia con datetime
            year_month_day = time.strftime("%Y-%m-%d", time.localtime())
            hour_minute_second = time.strftime("%H:%M:%S", time.localtime())

            # ------------------------------------
            key_user = f"train_started:user:{experiment_name}:{year_month_day}"
            user_info = {
                "gpu_ref": os.environ.get("WORKER_GPU_MODEL", None),
                "gpu_max_memory": os.environ.get("WORKER_GPU_MEMORY", None),
                "worker_ip": this_worker_ip,
                "time": hour_minute_second,
            }

            self.hash_manager.create_hash(
                key=key_user,
                hash_name=task_id,
                value=user_info,
                ttl=ttl_timeout,
            )

            # ------------------------------------
            worker_host = os.environ.get("WORKER_HOST", None)
            worker_user = os.environ.get("USER", None)
            key_worker = f"train_started:worker:{worker_host}_{worker_user}:{year_month_day}"
            worker_info = {
                "experiment_name": experiment_name,
                "user_code": user_code,
                "time": hour_minute_second,
            }

            self.hash_manager.create_hash(
                key=key_worker,
                hash_name=task_id,
                value=worker_info,
                ttl=ttl_timeout,
            )
        except Exception as e:
            logger.error(f"Error in train_notify: {e}")

    def execute_process(self, function: callable, args_dict: dict):
        self.start_time = time.time()
        results = None

        # Attempt to acquire the semaphore without blocking
        if self.semaphore.acquire(blocking=False):
            if self.resource_available:
                try:
                    # Mark the resource as unavailable
                    self.resource_available = False

                    self.notify_training_has_started(args_dict)

                    results = function(args_dict)
                except:
                    # Mark the resource as available again
                    self.resource_available = True
                    self.semaphore.release()  # Release the semaphore
                    raise
                finally:
                    # Mark the resource as available again
                    self.resource_available = True
                    self.semaphore.release()  # Release the semaphore
            else:
                self.semaphore.release()  # Release the semaphore if the resource was not available

        else:
            logger.info(
                f"Train: Could not acquire semaphore, will try again in 30 seconds."
            )

        self.end_time = time.time()
        return results
