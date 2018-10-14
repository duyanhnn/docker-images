#!/usr/bin/env bash

docker volume create taskforce_mssql_volume
docker-compose --project-name taskforce up -d