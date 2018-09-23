FROM neo4j:3.4.4
MAINTAINER Duy Anh <duyanhnguyenngoc97@gmail.com>

ADD init.sh /init.sh
ADD ./dump /

ENV NEO4J_AUTH=neo4j/12345

#ENV http_proxy=
#ENV https_proxy=

#RUN \
# chmod +x /init.sh && \
# apt-get update && apt-get dist-upgrade -y && \
# apt-get install psmisc -y -q && \
# apt-get autoremove -y && apt-get clean && \
# rm -rf /var/cache/* && rm -rf /var/lib/apt/lists/*

#ENTRYPOINT ["/init.sh"]

EXPOSE 7474 7473 7687