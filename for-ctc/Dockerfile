FROM anapsix/alpine-java:8_jdk

MAINTAINER Duy Anh <duyanhnguyenngoc97@gmail.com>

COPY ./.mvn /app/.mvn
COPY ./src /app/src
COPY ./mvnw /app/mvnw
COPY ./pom.xml /app/pom.xml

VOLUME /app/src/main/resources/static/files

WORKDIR /app

RUN chmod +x mvnw

CMD ["./mvnw", "spring-boot:run", "-Dspring-boot.run.profiles=stag"]

EXPOSE 8088