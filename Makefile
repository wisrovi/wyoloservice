MAKEFLAGS += --always-make


# ---------------------------------------------- NETWORK ----------------------------------------------
create_network:
	docker network create --subnet=28.10.4.0/24 --gateway=28.10.4.1 --driver=bridge nfs_network



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

