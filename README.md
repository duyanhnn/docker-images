# docker-images
To grant execute permission for every *.sh file, cd to your root project folder and run the following command:

```
find ./ -type f -iname "*.sh" -exec chmod +x {} \;
```
If you are planning on deploying this project on an EC2 instance, 
I recommend Amazon Linux 2 or Ubuntu Server LTS and increase storage size to **> 10GB**, 
to set up environmenton these instances, you can get my shell scripts at:
```
git clone -b master https://github.com/duyanhnn/docker-images-and-set-up-env.git
```
navigate to the root directory and run with sudo.

for Ubuntu Server (I recommend _setUpEnvAmazonUbuntu.sh_):
```
sudo bash ./setUpEnvAmazonUbuntu.sh
```
then reboot your instances and you are good to go!

for Compute Engine:
```
sudo bash ./setUpEnvComputeEngineUbuntu.sh
```
for Compute Engine but Docker + Docker Compose only:
```
sudo bash ./setUpDockerComputeEngineUbuntu.sh
```