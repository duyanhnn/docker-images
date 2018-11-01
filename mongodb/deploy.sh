#!/usr/bin/env bash

docker volume create bc2fp_mongodb_db_volume
docker volume create bc2fp_mongodb_configdb_volume
docker-compose --project-name bc2fp up -d