#!/usr/bin/env bash

gcloud config set account "onlyu.citynow.gcp1@gmail.com"
gcloud config set project ctcDeployment
gcloud compute --project "ctcdeployment" ssh --zone "australia-southeast1-b" "instance-1"