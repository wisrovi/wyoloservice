# pypi

pip install --upgrade build

# generar dist
python setup.py sdist bdist_wheel

twine upload dist/*


# docker
sudo docker login
sudo docker push wisrovi/wyoloservice:mlflow
sudo docker push wisrovi/wyoloservice:api
sudo docker push wisrovi/wyoloservice:worker
sudo docker push wisrovi/wyoloservice:user



# para la api
make build_env 
sudo docker push wisrovi/wyoloservice_api:v1.1.0
sudo docker tag wisrovi/wyoloservice_api:v1.1.0 wisrovi/wyoloservice_api:latest
sudo docker push wisrovi/wyoloservice_api:latest



VERSION=v1.0.11

# para el worker
make build_services start_services
sudo docker push wisrovi/wyoloservice_worker:v1.0.12
sudo docker tag wisrovi/wyoloservice_worker:v1.0.12 wisrovi/wyoloservice_worker:latest
sudo docker push wisrovi/wyoloservice_worker:latest


