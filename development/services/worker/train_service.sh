#!/bin/bash

# train_service.sh

# Monta el sistema de archivos CIFS y luego ejecuta el comando train_service

# Función para mostrar el uso del script
show_usage() {
  echo "Uso: $0 --config <ruta_al_archivo_de_configuracion>"
  exit 1
}

# Inicializa la variable para el archivo de configuración
config_file=""

# Procesa los argumentos de línea de comandos
while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      if [ -n "$2" ]; then
        config_file="$2"
        shift 2 # Elimina --config y su valor
      else
        echo "Error: Se requiere un valor para --config"
        show_usage
      fi
      ;;
    *)
      echo "Error: Argumento desconocido: $1"
      show_usage
      ;;
  esac
done

# Verifica si se proporcionó un archivo de configuración
if [ -z "$config_file" ]; then
  echo "Error: No se proporcionó el argumento --config"
  show_usage
fi

# Monta el sistema de archivos CIFS
sh -c /usr/local/bin/mount-cifs.sh

# Verifica si el montaje fue exitoso (puedes añadir una comprobación más robusta)
if [ ! -d "/datasets" ]; then
  echo "Error: El montaje de /datasets falló. Verifica el script mount-cifs.sh."
  exit 1
fi

# Ejecuta el comando train_service con el archivo de configuración
train_service worker.py --config "$config_file"

exit 0