MAKEFLAGS += --always-make

# ---------------------------------------------- SAMBAs ----------------------------------------------
mount_dataset:
	@if [ -z "$(CONTROL_HOST)" ]; then \
			echo "Error: CONTROL_HOST must be specified."; \
			exit 1; \
	fi
	sudo mkdir -p /mnt/train_service_datasets
	if ! mountpoint -q /mnt/train_service_datasets; then sudo mount -t cifs //$${CONTROL_HOST}/shared /mnt/train_service_datasets -o username=wisrovi,password=wyoloservice,port=23445,file_mode=0777,dir_mode=0777,iocharset=utf8 || exit 1; else echo "/mnt/train_service_datasets is already mounted."; fi
	
mount_db:
	@if [ -z "$(CONTROL_HOST)" ]; then \
			echo "Error: CONTROL_HOST must be specified."; \
			exit 1; \
	fi
	sudo mkdir -p /mnt/train_service_db
	if ! mountpoint -q /mnt/train_service_db; then sudo mount -t cifs //$${CONTROL_HOST}/shared /mnt/train_service_db -o username=wisrovi,password=wyoloservice,port=23448,file_mode=0777,dir_mode=0777,iocharset=utf8 || exit 1; else echo "/mnt/train_service_db is already mounted."; fi

mount_config_models:
	@if [ -z "$(CONTROL_HOST)" ]; then \
			echo "Error: CONTROL_HOST must be specified."; \
			exit 1; \
	fi
	sudo mkdir -p /mnt/train_service_config_models
	if ! mountpoint -q /mnt/train_service_config_models; then sudo mount -t cifs //$${CONTROL_HOST}/shared /mnt/train_service_config_models -o username=wisrovi,password=wyoloservice,port=23447,file_mode=0777,dir_mode=0777,iocharset=utf8 || exit 1; else echo "/mnt/train_service_config_models is already mounted."; fi


link_files_network: mount_dataset mount_db config_models

unlink_files_network:
	sudo umount /mnt/train_service_config_models || true
	sudo umount /mnt/train_service_datasets || true
	sudo umount /mnt/train_service_db || true

clean: unlink_files_network												




start_worker:
	docker-compose -f docker-compose.services.yml --compatibility up -d --build --force-recreate --no-deps