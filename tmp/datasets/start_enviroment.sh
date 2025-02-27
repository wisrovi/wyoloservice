docker run -it --rm --gpus all -w /datasets -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --shm-size=16g -v ./:/datasets ultralytics/ultralytics:8.2.45 bash
