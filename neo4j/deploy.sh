#!/usr/bin/env bash

docker volume create icop_neo4j_data_volume
docker volume create icop_neo4j_conf_volume
docker-compose --project-name icop up -d