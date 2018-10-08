#!/usr/bin/env bash

gcloud config set account "onlyu.year4sem1@gmail.com"
gcloud config set project y4s1-tlcn
gcloud compute --project "y4s1-tlcn" ssh --zone "asia-southeast1-b" "tlcn-instance-1"