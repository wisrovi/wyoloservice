  # docker run --rm --gpus all -it --env-file control_host.env --shm-size="24g" --privileged wisrovi/wyoloservice_worker:v1.0.10 zsh
  # docker run --rm --gpus all --env-file control_host.env --shm-size="24g" --privileged -it wisrovi/wyoloservice_worker:v1.0.10 ./train_service.sh --config /datasets/clasificacion/colorball.v8i.multiclass/config_train.yaml
