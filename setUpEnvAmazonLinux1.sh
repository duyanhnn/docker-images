#!/usr/bin/env bash

sudo yum install -y git
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
docker info