# wyoloservice (Legacy - Train Service 1)

> [!WARNING]
> **Deprecated:** This repository contains the legacy **Train Service 1** code. It has been completely superseded by [Train Service 2 (NeuralForgeAI)](file:///home/william.rodriguez/Documents/w_libraries/train_service2/wyoloservice2_production/README.md).
> Please refer to the new system for distributed optimizations, Celery queues, and native YOLO26 support.

---

## DOCKER IMAGES

https://hub.docker.com/r/wisrovi/wyoloservice/tags


## SERVICES

es importante que el ordenador a entrenar tenga al menos lo mismo de swap que de RAM

### aumentar swap


- Verifica si ya tienes swap:

    Abre una terminal y ejecuta el siguiente comando:

    ```bash
    sudo swapon --show
    ```
    Si ves alguna salida, significa que ya tienes swap configurado.

- Crea el archivo de intercambio:

    Utilizaremos el comando `fallocate` para crear un archivo de 32 GB. Ejecuta:

    ```bash
    sudo swapoff -a
    sudo fallocate -l 32G /swapfile
    ```

        

    Este proceso puede tardar un poco, dependiendo de la velocidad de tu disco duro.

- Establece los permisos correctos:

    Para mayor seguridad, establece los permisos del archivo de intercambio para que solo el usuario root pueda leerlo y escribir en él:

    ```bash
    sudo chmod 600 /swapfile
    ```

- Formatea el archivo como swap:

    Utiliza el comando `mkswap` para formatear el archivo como espacio de intercambio:

    ```bash
    sudo mkswap /swapfile
    ```

- Activa el swap:
    
    Activa el espacio de intercambio con el comando swapon:

    ```bash
    sudo swapon /swapfile
    ```

- Haz que el swap sea permanente:
    
    Para que el espacio de intercambio se active automáticamente al reiniciar el sistema, debes agregar una entrada al archivo `/etc/fstab`. Abre el archivo con un editor de texto (por ejemplo,`nano`):

    ```bash
    sudo nano /etc/fstab
    ```

    Agrega la siguiente línea al final del archivo:

    ```bash
    /swapfile swap swap defaults 0 0
    ```

    Guarda los cambios y cierra el editor.
    

- Ajusta la configuración de swappiness (opcional):

    El valor de "swappiness" controla con qué frecuencia el sistema operativo utiliza el swap. Un valor más bajo significa que el sistema intentará usar la RAM tanto como sea posible antes de recurrir al swap.

    Para ver el valor actual, ejecuta:

    ```bash
    cat /proc/sys/vm/swappiness
    ```

    Para cambiarlo, puedes editar el archivo `/etc/sysctl.conf`:

    ```bash
    sudo nano /etc/sysctl.conf
    ```

    Agrega o modifica la siguiente línea:

    ```bash
    vm.swappiness=10
    ```

    Guarda los cambios y ejecuta:

    ```bash
    sudo sysctl -p
    ```

- Verifica el swap:

    Para confirmar que el swap está configurado correctamente, ejecuta nuevamente:

    ```bash
    sudo swapon --show
    ```

    También puedes usar el comando `free -h` para ver el uso de la memoria RAM y el swap.

    * Consideraciones adicionales:

         Si tienes un disco SSD, es posible que desees configurar un valor de swappiness más bajo para reducir el desgaste del SSD.

         En sistemas con mucha RAM, es posible que no necesites tanto espacio de intercambio. Sin embargo, tener algo de swap puede ser útil en caso de que la RAM se llene por completo.

         Si estas usando un sistema de virtualización, es posible que la configuración del swap se realice desde el sistema anfitrión.


### montar los volumenes compartidos

- Se crean las carpetas que sincronizaran datos con los volumenes compartidos:

```
sudo mkdir -p /mnt/train_service_config_models
sudo mkdir -p /mnt/train_service_datasets
sudo mkdir -p /mnt/train_service_db
```


- por ejemplo: 
    - si el servidor donde se instalan los files y environment tiene la ip: 192.168.1.60


los volumenes se montan con:

```
sudo mount -t cifs //192.168.1.60/shared /mnt/train_service_datasets -o username=wisrovi,password=wyoloservice,port=23445,file_mode=0777,dir_mode=0777,iocharset=utf8

sudo mount -t cifs //192.168.1.60/shared /mnt/train_service_config_models -o username=wisrovi,password=wyoloservice,port=23447,file_mode=0777,dir_mode=0777,iocharset=utf8 

sudo mount -t cifs //192.168.1.60/shared /mnt/train_service_db -o username=wisrovi,password=wyoloservice,port=23448,file_mode=0777,dir_mode=0777,iocharset=utf8
```

Con esto si se escanea el contendido de las carpaetas creadas, ahora se podra ver contenido:

- para datasets:

    ```
    ls /mnt/train_service_datasets
    ```

- para database:

    ```
    ls /mnt/train_service_db
    ```


- para config_models:

   ```
    ls /mnt/train_service_config_models
    ```

Para automatizar los procesos se montan en el `crontad` para que en el inicio del SO se automonten:

- `crontad -e`

    ```
    @reboot sudo mount -t cifs //192.168.1.60/shared /mnt/train_service_datasets -o username=admin,password=admin,port=23445,file_mode=0777,dir_mode=0777,iocharset=utf8 >> /tmp/mount.log 2>&1

    @reboot sudo mount -t cifs //192.168.1.60/shared /mnt/train_service_config_models -o username=admin,password=admin,port=23447,file_mode=0777,dir_mode=0777,iocharset=utf8 >> /tmp/mount.log 2>&1

    @reboot sudo mount -t cifs //192.168.1.60/shared /mnt/train_service_db -o username=admin,password=admin,port=23448,file_mode=0777,dir_mode=0777,iocharset=utf8 >> /tmp/mount.log 2>&1
    ```

### Control server

El modo de funcionamiento de los servicios es el siguiente:

- Hay tres tipos de servicios: 
    - el servicio de entrenamiento
    - el servicio de control.
    - el servicio para que el usuario ponga el dataset.

- El servicio de entrenamiento se encarga de entrenar los modelos de YOLO, de estos puede haber varios.
- El servicio de control se encarga de registrar las peticones de entrenamiento y almacenar los resultados del entrenamiento.
- La forma en que se comunica el servidor de control con el servidor de entrenamiento es mediante un protocolo samba con 3 volumenes compartidos:
    - `train_service_datasets`
    - `train_service_config_models`
    - `train_service_db`
- El servicio de control se encarga de enviar las tareas de entrenamiento al servicio de entrenamiento, la forma de enviar las tareas es mediante un archivo yaml y el compartido del dataset, el archivo de configuracion y la base de datos para almacenar los resultados.


- El servicio de entrenamiento se ejecuta en el servidor de entrenamiento.
- El servicio de control se ejecuta en el servidor de control.
- El servicio de control se encarga de enviar las tareas de entrenamiento al servidor de entrenamiento.
- El servidor de entrenamiento se encarga de realizar el entrenamiento y enviar los resultados al servidor de control.
- El servidor de control se encarga de almacenar los resultados del entrenamiento en la base de datos.
- El servidor de control se encarga de enviar las configuraciones de entrenamiento al servidor de entrenamiento.
- El servidor de entrenamiento se encarga de utilizar las configuraciones de entrenamiento para realizar el entrenamiento.
- El servidor de control se encarga de almacenar los modelos entrenados en el servidor de entrenamiento en el servidor de control.
- Pueden haber varios servidores de entrenamiento y un solo servidor de control.


Para crear un servidor de control



### API

Para que la api funcione, perimero se debe crear una red de docker:

```bash
docker network create --subnet=28.10.4.0/24 --gateway=28.10.4.1 --driver=bridge nfs_network
```






```bash
URL=http://localhost:8000/docs
```

## MLFLOW

### minio

```bash
URL_UI=http://localhost:7450
URL_DATA=http://localhost:7449
USER=mlflow
PASSWORD=wyoloservice
```


### DVC


```bash
URL_UI=http://localhost:7452
URL_DATA=http://localhost:7451
USER=dvc
PASSWORD=wyoloservice
```

### Redis


```bash
URL_UI=http://localhost:7448
USER=root
PASSWORD=qwerty
```

### MLFLOW UI

```bash
URL_UI=http://localhost:7453
```

### POSTGRES


```bash
USER=postgres
PASSWORD=postgres
DB=wyoloservice
PORT=7454
```

### PGADMIN


```bash
URL=http://localhost:7455
EMAIL: wisrovi.rodriguez@gmail.com
PASSWORD: 12345678
```

Configurar asi:

```
    # General:
    #     name: postgres
    # conection:
    #     host name/address: postgres
    #     port: 5432
    #     username: postgres
    #     password: postgres
```


### NFS

```bash
PORT=7446
```

### SAMBA

```bash
URL=smb://localhost:23445/shared
USER=wisrovi
PASSWORD=wyoloservice
```
