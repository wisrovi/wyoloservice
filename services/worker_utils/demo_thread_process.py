import threading
import multiprocessing
import time

def train_yolo(request_config: dict, trial_number: int) -> dict:
    """Simula el entrenamiento de YOLO."""
    time.sleep(2)  # Simula el tiempo de entrenamiento
    return {"result": "success", "trial": trial_number}

def optuna_worker(q: multiprocessing.Queue, response_q: multiprocessing.Queue):
    """Hilo que coloca datos en la cola y espera respuesta."""
    for i in range(5):  # Simula la inserción de datos
        data = ({"param1": "value1", "param2": "value2"}, i)
        q.put(data)
        response = response_q.get()  # Espera la respuesta del scheduller_controller
        print(f"Respuesta recibida por el hilo: {response}")
        time.sleep(1)  # Simula tiempo entre tareas

def scheduller_controller(q: multiprocessing.Queue, response_q: multiprocessing.Queue):
    """Proceso que gestiona la cola y ejecuta train_yolo."""
    while True:
        request_config, trial_number = q.get()  # Espera un dato
        if isinstance(request_config, dict) and isinstance(trial_number, int):
            print("Nuevo dato recibido.")
            result = train_yolo(request_config, trial_number)
            result["status"] = "OK"
            response_q.put(result)  # Envía la respuesta al hilo optuna_worker
        else:
            print("Formato de datos incorrecto.")

if __name__ == "__main__":
    q = multiprocessing.Queue()
    response_q = multiprocessing.Queue()

    # Iniciar el worker en un hilo
    worker_thread = threading.Thread(target=optuna_worker, args=(q, response_q))
    worker_thread.start()

    # Iniciar el scheduler en un proceso
    scheduler_process = multiprocessing.Process(target=scheduller_controller, args=(q, response_q))
    scheduler_process.start()

    worker_thread.join()
    scheduler_process.terminate()  # Detiene el proceso después de que el hilo termine
