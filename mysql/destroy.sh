#!/usr/bin/env bash

docker-compose --project-name cnpmm down
docker rmi $(docker images --format '{{.Repository}}' | grep 'cnpmm_mysql_database')