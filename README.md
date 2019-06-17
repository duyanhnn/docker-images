# Set Up Environment For Development And Production Deployment Servers
To grant execute permission for every *.sh file, cd to your root project folder and run the following command:

```
find ./ -type f -iname "*.sh" -exec chmod +x {} \;
```
If you are planning on deploying this project on an EC2 instance, 
I recommend Amazon Linux 2 or Ubuntu Server LTS and increase storage size to **> 10GB**, 
To set up environmenton these instances, you can get my shell scripts at:
```
git clone -b master https://github.com/duyanhnn/docker-images-and-set-up-env.git
```
Navigate to the root directory and run with sudo.

For Ubuntu Server (I recommend _setUpEnvAmazonUbuntu.sh_):
```
sudo bash ./setUpEnvAmazonUbuntu.sh
```
Then reboot your instances and you are good to go!

For Compute Engine:
```
sudo bash ./setUpEnvComputeEngineUbuntu.sh
```
For Compute Engine but Docker + Docker Compose only:
```
sudo bash ./setUpDockerComputeEngineUbuntu.sh
```
For development environment only (Java + Maven):
```
sudo bash ./setUpDevelopmentEnv.sh
```