# enlazar la red de archivos:



```bash
make mount_dataset CONTROL_HOST=192.168.1.137
make mount_db CONTROL_HOST=192.168.1.137
make mount_config_models CONTROL_HOST=192.168.1.137
```

 ó

```bash
make link_files_network CONTROL_HOST=192.168.1.137
```

Se recomienda automaizar la creación de enlaces simbólicos con el siguiente comando:

```bash
sudo cronjob -e
```

```bash
@reboot /bin/bash /home/usuario/Documentos/Proyectos/production/2\ train_Server/make mount_dataset CONTROL_HOST=192.168.1.137
@reboot /bin/bash /home/usuario/Documentos/Proyectos/production/2\ train_Server/make mount_db CONTROL_HOST=192.168.1.137
@reboot /bin/bash /home/usuario/Documentos/Proyectos/production/2\ train_Server/make mount_config_models CONTROL_HOST=192.168.1.137
```





