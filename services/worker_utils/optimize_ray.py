
# import ray
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.search.optuna import OptunaSearch













def process_train_with_ray_tuner(request_config_user, task_id):
    """
    Realiza la optimización de hiperparámetros utilizando Ray Tune.

    Args:
        request_config_user (dict): Configuración proporcionada por el usuario.
        task_id (str): Identificador de la tarea.

    Returns:
        tuple: Mejor trial, ruta del mejor modelo y ruta de resultados.
    """
    sweeper_config = request_config_user.get("sweeper", {})

    # Espacio de búsqueda de hiperparámetros
    search_space = sweeper_config.get("search_space", {})

    # Convertir el espacio de búsqueda a un formato compatible con Ray Tune
    search_space = get_ray_suggestions(None, sweeper_config)

    @load_train_config(config_path=request_config_user.get("config_path"))
    def objective(config):
        """
        Función objetivo que entrena el modelo y reporta la métrica.

        Args:
            config (dict): Conjunto de hiperparámetros a probar.
        """
        try:
            # Actualizar la configuración con el task_id si es necesario
            config["task_id"] = task_id

            # Entrenar el modelo con los hiperparámetros actuales
            train_result = train_yolo(config)

            # Validar el resultado del entrenamiento
            if not isinstance(train_result, dict) or math.isnan(
                train_result.get("metric", float("nan"))
            ):
                raise ValueError("Rendimiento insuficiente.")

            # Obtener la métrica principal
            metric = train_result.get("metric", float("nan"))

            # Reportar la métrica a Ray Tune
            tune.report(metric=metric)

        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            traceback.print_exc()
            tune.report(
                metric=float("inf")
            )  # Reportar un valor alto para descartar este trial

    # Inicializar Ray
    ray.init(ignore_reinit_error=True)

    MODE = {
        "minimize": "min",
        "maximize": "max",
    }

    try:
        mode = MODE[sweeper_config.get("direction", "minimize")]

        # Configurar el scheduler (por ejemplo, ASHA para detener trials poco prometedores)
        scheduler = ASHAScheduler(
            metric="metric",  # Métrica a optimizar
            mode=mode,  # Dirección de optimización
            max_t=sweeper_config.get(
                "max_t", 100
            ),  # Número máximo de epochs o iteraciones
            grace_period=sweeper_config.get(
                "grace_period", 5
            ),  # Período mínimo antes de detener un trial
        )

        # Configurar el algoritmo de búsqueda (usando OptunaSearch como ejemplo)
        search_alg = OptunaSearch(
            metric="metric",
            mode=mode,
            space=search_space,
        )
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=sweeper_config.get("max_concurrent", 4)
        )

        # Ejecutar la optimización
        # TODO: revisar, aun no se logra ejecutar correctamente el tune.run, no entra a la funcion objetivo
        analysis = tune.run(
            objective,
            config=search_space,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=sweeper_config.get("n_trials", 1),
            resources_per_trial={
                "cpu": sweeper_config.get("cpu_per_trial", 1),
                "gpu": sweeper_config.get("gpu_per_trial", 0),
            },
        )

        # Obtener el mejor trial
        best_trial = analysis.get_best_trial(
            "metric", sweeper_config.get("direction", "minimize")
        )

        # Construir las rutas de resultados
        result_path = f'/models/{sweeper_config.get("study_name", "default_study")}/{request_config_user["type"]}/{request_config_user["task_id"]}'
        best_model_path = f"{result_path}/trail_history/trial_{best_trial.trial_id}.pt"

        return best_trial, best_model_path, result_path

    except Exception as e:
        logger.error(f"❌ Error al procesar el entrenamiento: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        # Asegurarse de detener Ray al finalizar
        ray.shutdown()

