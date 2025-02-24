from wredis.queue import RedisQueueManager

TOPIC = "training_queue"


queue_manager = RedisQueueManager(host="redis")


def enqueue_task(data_dict: dict):
    """Agregar tarea a la cola en Redis."""

    queue_manager.publish(TOPIC, data_dict)
