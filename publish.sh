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