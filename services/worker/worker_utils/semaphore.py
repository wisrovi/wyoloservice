import threading
import time
from loguru import logger


class SharedResource:
    def __init__(self):
        self.semaphore = threading.Semaphore(1)  # Initialized to 1, acts like a lock
        self.resource_available = True
        self.end_time = 0
        self.start_time = 0

    def elapsedtime(self):
        return self.end_time - self.start_time

    def execute_process(self, function: callable, args_dict: dict):
        self.start_time = time.time()
        results = None

        thread_name = "function"
        if self.semaphore.acquire(
            blocking=False
        ):  # Attempt to acquire the semaphore without blocking
            if self.resource_available:

                try:
                    self.resource_available = False  # Mark the resource as unavailable
                    logger.info(
                        f"Thread {thread_name}: Resource acquired, executing process..."
                    )

                    results = function(args_dict)
                finally:
                    self.resource_available = (
                        True  # Mark the resource as available again
                    )
                    self.semaphore.release()  # Release the semaphore

            else:
                self.semaphore.release()  # Release the semaphore if the resource was not available
                logger.info(
                    f"Thread {thread_name}: Resource busy, will try again in 30 seconds."
                )

        else:
            logger.info(
                f"Thread {thread_name}: Could not acquire semaphore, will try again in 30 seconds."
            )

        self.end_time = time.time()
        return results
