#!/usr/bin/env bash

docker-compose -f docker-compose-dev.yml --project-name cinamon down
docker rmi $(docker images --format '{{.Repository}}' | grep 'cinamon_mysql')
docker rmi $(docker images --format '{{.Repository}}' | grep 'cinamon_redis')
docker rmi $(docker images --format '{{.Repository}}' | grep 'cinamon_app')