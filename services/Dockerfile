FROM ultralytics/ultralytics:8.3.20


USER root


# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install net-tools
RUN apt install iputils-ping -y

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# ZSH
RUN apt-get update && apt-get install -y zsh
RUN apt-get install -y wget
# Uses "robbyrussell" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t aussiegeek
# customizations
RUN apt-get install figlet -y
RUN echo "alias ll='ls -l'" >> ~/.zshrc



# Limpiar caché y archivos temporales
RUN rm -rf /var/cache/apk/*
RUN rm -rf /tmp/*
RUN apt-get clean




# Configurar el entorno de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar wait-for-it.sh al contenedor
COPY wait-for-it.sh /usr/local/bin/wait-for-it.sh
RUN chmod +x /usr/local/bin/wait-for-it.sh


# Copiar el código fuente
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000


RUN echo "figlet wyoloservice" >> ~/.zshrc
RUN echo "figlet wyoloservice" >> ~/.bashrc


# Comando por defecto
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Comando por defecto
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
