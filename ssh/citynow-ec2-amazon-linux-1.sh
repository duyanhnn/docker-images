#!/usr/bin/env bash

chmod 400 books_staging.pem
ssh -i books_staging.pem ec2-user@ec2-13-113-191-82.ap-northeast-1.compute.amazonaws.com