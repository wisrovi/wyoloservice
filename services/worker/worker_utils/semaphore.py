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

        # Attempt to acquire the semaphore without blocking
        if self.semaphore.acquire(blocking=False):
            if self.resource_available:
                try:
                    # Mark the resource as unavailable
                    self.resource_available = False

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
