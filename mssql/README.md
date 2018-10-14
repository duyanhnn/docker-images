# Database (only) Deployment Instructions
Configurations for Docker (deployment) of Task Force Project, and this README.md was written by Duy Anh Nguyễn Ngọc.

To deploy MS SQL Server database to any environment just simply clone the project:
```
git clone -b master https://github.com/duyanhnn/docker-images.git
```
Navigate to the root directory -> mssql-linux folder and run:
```
./deployDatabase.sh
```
Or run with sudo:
```
sudo bash deployDatabase.sh
```
Make sure that deploy environment has installed Docker + Docker Compose.

If you want to shut down (warning! will erase images):
```
./destroyDatabase.sh
```
Or run with sudo:
```
sudo bash destroyDatabase.sh
```