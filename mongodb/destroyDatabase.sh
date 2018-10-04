#!/usr/bin/env bash

docker-compose --project-name bc2fp down
docker rmi $(docker images --format '{{.Repository}}' | grep 'bc2fp_mongodb_database')