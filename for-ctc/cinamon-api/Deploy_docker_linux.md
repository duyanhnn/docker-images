# Flax CTC Docker Guideline

# Operating System
Ubuntu 16.04

If your system doesn't have Docker yet:
### Install Docker
```
$ sudo apt-get update
$ sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    python-pip \
    unzip
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable" -y
$ sudo apt-get update
$ sudo apt-get install docker-ce -y
```

Ref link: https://docs.docker.com/install/linux/docker-ce/ubuntu/


1. Install docker-compose
```
$ sudo pip install docker-compose
```

2. Create data directories
```
$ sudo mkdir /data
$ sudo mkdir /debug
```
3. Stop MySQL and Redis if they are running:
Stop MySQL
```
$ sudo systemctl stop mysql
```

Stop Redis
```
$ sudo systemctl stop redis-server
```

4. Run
In source code folder
```
$ sudo docker-compose -f docker-compose-dev.yml up -d
```

Note: check containers running
```
$ sudo docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                    NAMES
aff20973ef1e        prj_flax_ctc_2_app  "/bin/bash /code/ent…"   2 hours ago         Up 2 hours          0.0.0.0:8000->8000/tcp   prj_flax_ctc2_app_1
e30ac18ca249        mysql:5.7           "docker-entrypoint.s…"   2 hours ago         Up 2 hours          0.0.0.0:3306->3306/tcp   prj_flax_ctc2_mysql_1
b24749203474        redis               "docker-entrypoint.s…"   2 hours ago         Up 2 hours          0.0.0.0:6379->6379/tcp   prj_flax_ctc2_redis_1
```


### Stop
```
$ sudo docker-compose -f docker-compose-dev.yml stop
```

### Logs
- uwsgi's log: /var/log/uwsgi.log
- celery's log: /var/log/celery.log
- supervisord's log: /var/log/supervisord.log


### Notes
1, If you want to change expose port of API (default 8000), change this line in docker-compose-dev.yml
```
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"  <-- this line
```

Example change to port 80: `- "80:8000"`
Stop and Start the docker-compose-dev.yml so change will take effect.