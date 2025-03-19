#!/bin/bash

# Imprimir variables de entorno para depuración
echo "CONTROL_HOST: $CONTROL_HOST"
echo "CIFS_USER: $CIFS_USER"
echo "CIFS_PASS: $CIFS_PASS"

# Leer variables de entorno (con valores predeterminados)
CONTROL_HOST=${CONTROL_HOST:-localhost}
CIFS_USER=${CIFS_USER:-wisrovi}
CIFS_PASS=${CIFS_PASS:-wyoloservice}

# Montar las unidades CIFS
echo "Montando /config_versions..."
mount -t cifs //$CONTROL_HOST/shared /config_versions \
    -o username=$CIFS_USER,password=$CIFS_PASS,port=23447,file_mode=0777,dir_mode=0777,iocharset=utf8

echo "Montando /database..."
mount -t cifs //$CONTROL_HOST/shared /database \
    -o username=$CIFS_USER,password=$CIFS_PASS,port=23448,file_mode=0777,dir_mode=0777,iocharset=utf8

echo "Montando /datasets/..."
mount -t cifs //$CONTROL_HOST/shared /datasets/ \
    -o username=$CIFS_USER,password=$CIFS_PASS,port=23445,file_mode=0777,dir_mode=0777,iocharset=utf8