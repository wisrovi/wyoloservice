from wredis.streams import RedisStreamManager


def main(task_id):
    stream_manager = RedisStreamManager(
        host="192.168.1.60",
        verbose=False,
        port=23438,
    )

    print("Streaming activated in", f"stream:{task_id}")

    @stream_manager.on_message(
        stream_name=f"stream:{task_id}",
        group_name=f"stream:{task_id}",
        consumer_name=f"stream:{task_id}",
    )
    def process_message(data):
        print(data.get("value"))

    # Mantener el programa activo para consumir mensajes
    stream_manager.wait()


if __name__ == "__main__":
    main("7733ece2ceaa4a67884b202c65fc9904")
