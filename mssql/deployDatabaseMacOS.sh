#!/usr/bin/env bash

docker volume create taskforce_mssql_volume
docker-compose -f docker-compose-macos.yml --project-name taskforce up -d