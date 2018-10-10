#!/usr/bin/env bash

docker-compose --project-name taskforce down
docker rmi $(docker images --format '{{.Repository}}' | grep 'taskforce_mssql_database')