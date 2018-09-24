#!/usr/bin/env bash

echo 'Setting up temp directory...'
mkdir ~/tmp
sudo echo 'export TMPDIR=$HOME/tmp' >> ~/.bash_profile
source ~/.bash_profile
echo 'Done setting up temp directory'
echo 'Setting up locale and language.'
sudo echo 'export LANGUAGE=en_US.UTF-8' >> ~/.bash_profile
sudo echo 'export LC_ALL=en_US.UTF-8' >> ~/.bash_profile
sudo echo 'export LANG=en_US.UTF-8' >> ~/.bash_profile
sudo echo 'export LC_TYPE=en_US.UTF-8' >> ~/.bash_profile
source ~/.bash_profile
echo 'Done setting up locale and language.'
echo 'Setting up swap memory, this might take a long time, please wait...'
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=3072
sudo /sbin/mkswap /var/swap.1
sudo chmod 600 /var/swap.1
sudo /sbin/swapon /var/swap.1
sudo echo '' >> /etc/fstab
sudo echo '/var/swap.1	swap	swap	defaults		0	0' >> /etc/fstab
free -m
echo 'Done setting up swap memory, if /etc/fstab was not setted, you have to set it manually! Please see log.'
sudo apt-get update -y
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update -y
sudo apt-get install -y docker-ce
sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
sudo groupadd docker
sudo gpasswd -a ubuntu docker
git clone -b docker/develop https://gitlab.com/citynow/ICOP-Project-Backend.git icop-docker-develop
sudo chmod -R a+rwx icop-docker-develop