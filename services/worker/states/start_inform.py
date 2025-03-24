import os
from wredis.hash import RedisHashManager
import yaml


class Start_inform:
    
    __NAME__ = "start_inform"
    __VERSION__ = "v1.0"

    def __call__(self, args_dict: dict):

        config_path = args_dict.get("config_path", None)
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        redis_config = config.get("redis", None)
        if redis_config is None:
            return {}

        hash_manager = RedisHashManager(
            host=redis_config.get("REDIS_HOST"),
            port=redis_config.get("REDIS_PORT"),
            db=redis_config.get("REDIS_DB"),
            verbose=False,
        )

        metadata = {
            other_metadata: os.environ.get(other_metadata, None)
            for other_metadata in self.worker_metadata
        }

        trial_number = config.get("trial_number", 1)
        total_trails = config.get("sweeper", 1).get("n_trials", 10)

        total_epochs = config.get("train", {}).get("epochs", 10)

        task_id = config.get("task_id", "noTaskId")

        metadata["TRIAL_NUMBER"] = trial_number
        metadata["TOTAL_TRIALS"] = total_trails
        metadata["TOTAL_EPOCHS"] = total_epochs
        metadata["EPOCH"] = 0
        metadata["EPOCH_PROGRESS"] = 0
        metadata["TRIAL_PROGRESS"] = trial_number / total_trails

        redis_key = "progress" + f":{task_id}"

        for metadata_key, metadata_value in metadata.items():
            hash_manager.create_hash(
                key=redis_key,
                hash_name=metadata_key,
                value=metadata_value,
                ttl=120,
            )

        return {}
