from wredis.queue import RedisQueueManager

TOPIC = "training_queue"


queue_manager = RedisQueueManager(host="redis")


def send_task_to_worker(data_dict: dict):
    """Agregar tarea a la cola en Redis."""

    queue_manager.publish(TOPIC, data_dict)
