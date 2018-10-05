#!/usr/bin/env bash

gcloud config set account `onlyu.citynow.gcp1@gmail.com`
gcloud config set project fluted-quasar-218508
gcloud compute --project "fluted-quasar-218508" ssh --zone "asia-southeast1-b" "ctc-instance-1"