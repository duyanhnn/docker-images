# Database (only) Deployment Instructions
Configurations for Docker (deployment) of ICOP Project, and this README.md was written by Duy Anh Nguyễn Ngọc.

To deploy Neo4j database to any environment just simply clone the project at branch docker/develop with command:
```
git clone -b develop https://gitlab.com/citynow/BC2FProject-BackEnd.git bc2fp-develop
```
Navigate to the root directory -> neo4j folder and run:
```
./deployDatabase.sh
```
Or run with sudo:
```
sudo ./deployDatabase.sh
```
Make sure that deploy environment has installed Docker + Docker Compose.

If you want to shut down (warning! will erase images), run (or with sudo):
```
./destroyDatabase.sh
```