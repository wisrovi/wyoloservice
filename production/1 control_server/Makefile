MAKEFLAGS += --always-make


# ---------------------------------------------- NETWORK ----------------------------------------------
create_network:
	docker network create --subnet=28.10.4.0/24 --gateway=28.10.4.1 --driver=bridge nfs_network

config.py:
	pip install customtkinter
	curl -o config.py https://raw.githubusercontent.com/wisrovi/wyoloservice/refs/heads/main/production/1%20control_server/config.py

docker-compose.files.yml:
	curl -o docker-compose.files.yml https://raw.githubusercontent.com/wisrovi/wyoloservice/refs/heads/main/production/1%20control_server/docker-compose.files.yml

docker-compose.environment.yml:
	curl -o docker-compose.environment.yml https://raw.githubusercontent.com/wisrovi/wyoloservice/refs/heads/main/production/1%20control_server/docker-compose.environment.yml

start_control_server: config.py docker-compose.files.yml docker-compose.environment.yml
# create environment
	docker-compose -f docker-compose.environment.yml  up -d --build --force-recreate --no-deps
# create files system
	docker-compose -f docker-compose.files.yml  up -d --build --force-recreate --no-deps

certs:
	mkdir -p ./nginx/certs/www.wyoloservice.com
	docker-compose -f docker-compose.proxy.yaml up -d --build --force-recreate --no-deps
	

start_control_server_proxy: start_control_server certs