#!/usr/bin/env bash

chmod 400 books_production_key.pem
sudo ssh -i books_production_key.pem ubuntu@54.168.100.61
#sudo ssh ubuntu@54.168.100.61
#sudo ssh ec2-user@54.168.100.61
#sudo ssh root@54.168.100.61
#sudo ssh fedora@54.168.100.61
#sudo ssh admin@54.168.100.61

# check $HOME/.ssh/authorized_keys
# https://askubuntu.com/questions/466549/bash-home-user-ssh-authorized-keys-no-such-file-or-directory
# https://stackoverflow.com/questions/18551556/permission-denied-publickey-when-ssh-access-to-amazon-ec2-instance
#
