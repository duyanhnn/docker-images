#!/usr/bin/env bash

echo 'Setting up swap memory.'
sudo mount /dev/xvda2 /mnt
sudo dd if=/dev/zero of=/mnt/swapfile bs=1M count=5120
sudo chown root:root /mnt/swapfile
sudo chmod 600 /mnt/swapfile
sudo mkswap /mnt/swapfile
sudo swapon /mnt/swapfile
df -T
sudo echo '' >> /etc/fstab
sudo echo '/dev/xvda1	/mnt	auto	defaults,nobootwait,comment=cloudconfig		0	2' >> /etc/fstab
sudo echo '/mnt/swapfile	swap	swap	defaults		0	0' >> /etc/fstab
sudo swapon -a
free -m
echo echo 'Done setting up swap memory.'
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