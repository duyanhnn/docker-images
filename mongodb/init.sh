#!/bin/bash

if test -z "$BC2FP_DATABASE_USERNAME"; then
    echo "BC2FP_DATABASE_USERNAME not defined"
    exit 1
fi

if test -z "$BC2FP_DATABASE_PASSWORD"; then
    echo "BC2FP_DATABASE_PASSWORD not defined"
    exit 1
fi

auth="-u $BC2FP_DATABASE_USERNAME -p $BC2FP_DATABASE_USERNAME"

# MONGODB USER CREATION
(
echo "setup mongodb auth"
create_user="if (!db.getUser('$BC2FP_DATABASE_USERNAME')) { db.createUser({ user: '$BC2FP_DATABASE_USERNAME', pwd: '$BC2FP_DATABASE_PASSWORD', roles: [ {role:'readWrite', db:'bc2fp'} ]}) }"
until mongo bc2fp --eval "$create_user" || mongo bc2fp $auth --eval "$create_user"; do sleep 5; done
killall mongod
sleep 1
killall -9 mongod
) &

# INIT DUMP EXECUTION
(
if test -n "$INIT_DUMP"; then
    echo "execute dump file"
	until mongo bc2fp $auth $INIT_DUMP; do sleep 5; done
fi
) &

echo "start mongodb without auth"
chown -R mongodb /data/db
gosu mongodb mongod --bind_ip_all "$@"

echo "restarting with auth on"
sleep 5
exec gosu mongodb mongod --bind_ip_all --auth "$@"