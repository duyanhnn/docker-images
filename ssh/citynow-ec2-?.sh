#!/usr/bin/env bash

chmod 400 books_production_key.pem
ssh -i books_production_key.pem ubuntu@54.168.100.61
#ssh ec2-user@54.168.100.61
#ssh root@54.168.100.61
#ssh fedora@54.168.100.61
#ssh admin@54.168.100.61