# docker-images
If you are planning on deploying this project on an EC2 instance, 
I recommend Amazon Linux 2 or Ubuntu Server LTS and increase storage size to **> 10GB**, 
to set up environmenton these instances, you can get my shell scripts at:
```
git clone -b master https://github.com/duyanhnn/docker-images.git
```
navigate to the root directory and run with sudo.

for Ubuntu Server (I recommend _setUpEnvUbuntu2.sh_):
```
sudo bash ./setUpEnvUbuntu2.sh
```
or
```
sudo bash ./setUpEnvUbuntu.sh
```
then reboot your instances and you are good to go!

for Compute Engine:
```
sudo bash ./setUpEnvUbuntuComputeEngine.sh
```
