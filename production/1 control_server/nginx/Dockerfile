# Usamos Debian como base para instalar OpenSSL
FROM debian:bullseye-slim

# Instalamos OpenSSL
RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

# Definimos el directorio de trabajo donde OpenSSL guardará los certificados
WORKDIR /data
