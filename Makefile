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



start_services:
	docker-compose -f docker-compose.services.yml --compatibility up -d --build --force-recreate --no-deps

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