MAKEFLAGS += --always-make


# ---------------------------------------------- NETWORK ----------------------------------------------
create_network:
	docker network create --subnet=28.10.4.0/24 --gateway=28.10.4.1 --driver=bridge nfs_network


create_files_server:
	docker-compose -f docker-compose.files.yml --compatibility up -d --build --force-recreate --no-deps


create_control_server:
	docker-compose -f docker-compose.environment.yml --compatibility up -d --build --force-recreate --no-deps

get_ip:
	echo CONTROL_HOST=$$(hostname -I | awk '{print $$1}') > control_host.env
