#!/usr/bin/env bash

if [ ! -d "./data/base" ]; then
    echo "base folder not existed! Creating..."
    mkdir -p ./data/base
    echo "done!"
else
    echo "base folder existed!"
fi

if [ ! -d "./data/global" ]; then
    echo "global folder not existed! Creating..."
    mkdir -p ./data/global
    echo "done!"
else
    echo "global folder existed!"
fi

if [ ! -d "./data/pg_commit_ts" ]; then
    echo "pg_commit_ts folder not existed! Creating..."
    mkdir -p ./data/pg_commit_ts
    echo "done!"
else
    echo "pg_commit_ts folder existed!"
fi

if [ ! -d "./data/pg_dynshmem" ]; then
    echo "pg_dynshmem folder not existed! Creating..."
    mkdir -p ./data/pg_dynshmem
    echo "done!"
else
    echo "pg_dynshmem folder existed!"
fi

if [ ! -d "./data/pg_logical" ]; then
    echo "pg_logical folder not existed! Creating..."
    mkdir -p ./data/pg_logical
    echo "done!"
else
    echo "pg_logical folder existed!"
fi

if [ ! -d "./data/pg_logical/mappings" ]; then
    echo "mappings folder not existed! Creating..."
    mkdir -p ./postgres/data/pg_logical/mappings
    echo "done!"
else
    echo "mappings folder existed!"
fi

if [ ! -d "./data/pg_logical/snapshots" ]; then
    echo "snapshots folder not existed! Creating..."
    mkdir -p ./postgres/data/pg_logical/snapshots
    echo "done!"
else
    echo "snapshots folder existed!"
fi

if [ ! -d "./data/pg_multixact" ]; then
    echo "pg_multixact folder not existed! Creating..."
    mkdir -p ./data/pg_multixact
    echo "done!"
else
    echo "pg_multixact folder existed!"
fi

if [ ! -d "./data/pg_notify" ]; then
    echo "pg_notify folder not existed! Creating..."
    mkdir -p ./data/pg_notify
    echo "done!"
else
    echo "pg_notify folder existed!"
fi

if [ ! -d "./data/pg_replslot" ]; then
    echo "pg_replslot folder not existed! Creating..."
    mkdir -p ./data/pg_replslot
    echo "done!"
else
    echo "pg_replslot folder existed!"
fi

if [ ! -d "./data/pg_serial" ]; then
    echo "pg_serial folder not existed! Creating..."
    mkdir -p ./data/pg_serial
    echo "done!"
else
    echo "pg_serial folder existed!"
fi

if [ ! -d "./data/pg_snapshots" ]; then
    echo "pg_snapshots folder not existed! Creating..."
    mkdir -p ./data/pg_snapshots
    echo "done!"
else
    echo "pg_snapshots folder existed!"
fi

if [ ! -d "./data/pg_stat" ]; then
    echo "pg_stat folder not existed! Creating..."
    mkdir -p ./data/pg_stat
    echo "done!"
else
    echo "pg_stat folder existed!"
fi

if [ ! -d "./data/pg_stat_tmp" ]; then
    echo "pg_stat_tmp folder not existed! Creating..."
    mkdir -p ./data/pg_stat_tmp
    echo "done!"
else
    echo "pg_stat_tmp folder existed!"
fi

if [ ! -d "./data/pg_subtrans" ]; then
    echo "pg_subtrans folder not existed! Creating..."
    mkdir -p ./data/pg_subtrans
    echo "done!"
else
    echo "pg_subtrans folder existed!"
fi

if [ ! -d "./data/pg_tblspc" ]; then
    echo "pg_tblspc folder not existed! Creating..."
    mkdir -p ./data/pg_tblspc
    echo "done!"
else
    echo "pg_tblspc folder existed!"
fi

if [ ! -d "./data/pg_twophase" ]; then
    echo "pg_twophase folder not existed! Creating..."
    mkdir -p ./data/pg_twophase
    echo "done!"
else
    echo "pg_twophase folder existed!"
fi

if [ ! -d "./data/pg_wal" ]; then
    echo "pg_wal folder not existed! Creating..."
    mkdir -p ./data/pg_wal
    echo "done!"
else
    echo "pg_wal folder existed!"
fi

if [ ! -d "./data/pg_wal/archive_status" ]; then
    echo "archive_status folder not existed! Creating..."
    mkdir -p ./postgres/data/pg_wal/archive_status
    echo "done!"
else
    echo "archive_status folder existed!"
fi

if [ ! -d "./data/pg_xact" ]; then
    echo "pg_xact folder not existed! Creating..."
    mkdir -p ./data/pg_xact
    echo "done!"
else
    echo "pg_xact folder existed!"
fi

docker-compose --project-name ctc up -d