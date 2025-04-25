MAKEFLAGS += --always-make


# ---------------------------------------------- NETWORK ----------------------------------------------
create_network:
	docker network create --subnet=28.10.4.0/24 --gateway=28.10.4.1 --driver=bridge nfs_network



# ---------------------------------------------- FILES ----------------------------------------------
start_files:
	docker-compose -f docker-compose.files.yml --compatibility up -d --build --force-recreate --no-deps

stop_files:
	docker-compose -f docker-compose.files.yml down

build_files:
	docker-compose -f docker-compose.files.yml build



# ---------------------------------------------- ENVIRONMENT ----------------------------------------------
start_env:
	docker-compose -f docker-compose.environment.yml --compatibility up -d --build --force-recreate --no-deps

stop_env:
	docker-compose -f docker-compose.environment.yml down

build_env:
	docker-compose -f docker-compose.environment.yml build


# ---------------------------------------------- SERVICES ----------------------------------------------

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

start_services: user.env
	docker-compose -f docker-compose.services.yml --env-file user.env  --compatibility up -d --build --force-recreate --no-deps

stop_services:
	docker-compose -f docker-compose.services.yml down

build_services:
	docker-compose -f docker-compose.services.yml build

into_worker:
	docker-compose -f docker-compose.services.yml exec  worker bash

logs_worker:
	docker-compose -f docker-compose.services.yml logs -f  worker





# ---------------------------------------------- SERVICES ----------------------------------------------


start_user:
	docker-compose -f docker-compose.user.yml build --no-cache
	docker-compose -f docker-compose.user.yml --compatibility up -d --build --force-recreate --no-deps

into_user:
	docker-compose -f docker-compose.user.yml exec  user bash


stop_user:
	docker-compose -f docker-compose.user.yml down