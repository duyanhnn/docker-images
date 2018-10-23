#!/usr/bin/env bash

docker-compose --project-name ctc down
docker rmi $(docker images --format '{{.Repository}}' | grep 'ctc_postgres_database')
docker rmi $(docker images --format '{{.Repository}}' | grep 'ctc_cinamon_app')
docker rmi $(docker images --format '{{.Repository}}' | grep 'ctc_ctc_webapp')