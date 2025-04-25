MAKEFLAGS += --always-make

start_worker: config.py user.env docker-compose.services.yml
	python config.py
	docker-compose -f docker-compose.services.yml --env-file user.env  up -d --build  --force-recreate --no-deps --pull always

user.env:
	echo "USER=$$(whoami)" > user.env
	echo "WORKER_HOST=$$(hostname -I | awk '{print $$1}')" >> user.env
	echo "WORKER_HOSTNAME=$$(hostname)" >> user.env
	echo "WORKER_OS=$$(uname -s)" >> user.env
	echo "WORKER_KERNEL_VERSION=$$(uname -r)" >> user.env
	echo "WORKER_CPU_CORES=$$(nproc)" >> user.env
	echo "WORKER_GATEWAY=$$(ip route | grep default | awk '{print $$3}')" >> user.env
	echo "WORKER_NETWORK_INTERFACE=$$(ip route | grep default | awk '{print $$5}')" >> user.env
	echo "WORKER_DOCKER_VERSION=$$(docker --version | awk '{print $$3}' | sed 's/,//')" >> user.env
	echo "WORKER_APP_BASE_PATH=$$(pwd)" >> user.env
	echo "WORKER_APP_ENV=production" >> user.env
	echo "WORKER_HOME_DIR=$$HOME" >> user.env
	echo "WORKER_CURRENT_DATE=$$(date '+%Y-%m-%d')" >> user.env
	echo "WORKER_CURRENT_TIME=$$(date '+%H:%M:%S')" >> user.env
	echo "WORKER_GPU_COUNT=$$(nvidia-smi --query-gpu=count --format=csv,noheader)" >> user.env
	echo "WORKER_GPU_MODEL=$$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" >> user.env
	echo "WORKER_GPU_MEMORY=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)" >> user.env
	echo "WORKER_GPU_MEMORY=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)" >> user.env
	MEM_TOTAL=$$(awk '/^MemTotal:/ {print $$2}' /proc/meminfo); \
    echo "WORKER_RAM_MEMORY=$$((MEM_TOTAL / 1048576 * 8 / 10))g" >> user.env
	echo "WORKER_CPU_CORES=$$((`nproc` - 1)).0" >> user.env

docker-compose.services.yml:
	curl -o docker-compose.services.yml "https://raw.githubusercontent.com/wisrovi/wyoloservice/refs/heads/main/base/2%20train_Server/docker-compose.services.yml"

config.py:
	pip install customtkinter
	curl -o config.py "https://raw.githubusercontent.com/wisrovi/wyoloservice/refs/heads/main/base/2%20train_Server/config.py"

clean:
	rm -f user.env docker-compose.services.yml

log:
	docker-compose -f docker-compose.services.yml logs -f --tail 100

into:
	docker-compose -f docker-compose.services.yml exec worker zsh