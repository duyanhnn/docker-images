#!/bin/bash

#cd /code/ctc
#uwsgi --ini uwsgi.ini

if [ -e /var/run/supervisor.sock ]; then
    unlink /var/run/supervisor.sock
fi
/usr/bin/supervisord