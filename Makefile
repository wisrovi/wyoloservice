MAKEFLAGS += --always-make


# ---------------------------------------------- NETWORK ----------------------------------------------
create_network:
	docker network create --subnet=28.10.4.0/24 --gateway=28.10.4.1 --driver=bridge nfs_network



# ---------------------------------------------- ENVIRONMENT ----------------------------------------------
start_env:
	docker-compose -f docker-compose.environment.yml --compatibility up -d --build --force-recreate --no-deps

stop_env:
	docker-compose -f docker-compose.environment.yml down


# ---------------------------------------------- SERVICES ----------------------------------------------



start:
	docker-compose -f docker-compose.yml --compatibility up -d --build --force-recreate --no-deps

stop:
	docker-compose -f docker-compose.yml down

build:
	docker-compose -f docker-compose.yml build

debug:
	docker-compose -f docker-compose.yml exec web python3 app.py

into:
	docker-compose -f docker-compose.yml exec web zsh

log:
	docker-compose -f docker-compose.yml logs -f web