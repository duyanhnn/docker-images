#!/usr/bin/env bash

docker-compose --project-name icop down
docker rmi $(docker images --format '{{.Repository}}' | grep 'icop_neo4j_database')