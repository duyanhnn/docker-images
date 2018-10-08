#!/usr/bin/env bash

gcloud config set account "duyanhnguyenngoc16@gmail.com"
gcloud config set project onlyu-216914
gcloud compute --project "onlyu-216914" ssh --zone "asia-southeast1-a" "instance-1"