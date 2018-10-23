# Elasticsearch for Mizuho's project

1. Install elasticsearch
2. Import database to elasticsearch server
3. Run MizuhoElastic flow

## 1. Install Elasticsearch server

Firstly, you must update your apt-get

`sudo apt-get update`


Download .deb file to install elasticsearch on link
[Download Elasticsearch Free • Get Started Now       | Elastic](https://www.elastic.co/downloads/elasticsearch)

Install elasticsearch

`sudo dpkg -i elasticsearch*.deb`

To make sure Elasticsearch starts and stops automatically with the server, add its init script to the default runlevels

`sudo systemctl enable elasticsearch.service`


Now you can start Elasticsearch for the first time.

`sudo systemctl start elasticsearch`

<Optional> To restart Elasticsearch with the command
`sudo systemctl restart elasticsearch`

By now, Elasticsearch should be running on port 9200. You can test it with curl, the command line client-side URL transfers tool and a simple GET request.

`curl -X GET 'http://localhost:9200'`

Output of curl
```
{
  "name" : "My First Node",
  "cluster_name" : "mycluster1",
  "version" : {
    "number" : "2.3.1",
    "build_hash" : "bd980929010aef404e7cb0843e61d0665269fc39",
    "build_timestamp" : "2016-04-04T12:25:05Z",
    "build_snapshot" : false,
    "lucene_version" : "5.5.0"
  },
  "tagline" : "You Know, for Search"
}
```

## 2. Importer for Elasticsearch


Input: Path of a database with csv file format

To import database to Elastic Server

```py
from elastic_importer import ElasticImporter

importer = ElasticImporter('<csv_path>')
importer.process()

---> 'Imported the database of <csv_name> to Elasticsearch Server!'
```

**Note** If database has imported to Elasticsearch, you will not need to run it again. Or if you must to re-import with a database edited, you must delete old database with a command, that is `curl -XDELETE localhost:9200/<csv_name_old>`    


## 3. Run MizuhoElastic flow

**How to use**

```py
# input_json template
{
    'company_name': '',
    'company_address':'',
    'tel':'',
    'fax':'',
    'postcode':'',
    'bank1': {
                 'bank': '',
                 'branch': '',
                 'type_of_account': '',
                 'account_number': '',
	               'account_name':''
             },
    'bank2': {
                 'bank': '',
                 'branch': '',
                 'type_of_account': '',
                 'account_number': '',
                 'account_name':''
             },
    ...
}

# output_json template
{
    'company_name': '',
    'company_address':'',
    'tel':'',
    'fax':'',
	  'postcode':'',
    'bank_database1': {
                 'bank': '',
                 'branch': '',
                 'type_of_account': '',
                 'account_number': '',
                 'account_name':''
             },
    'bank_database2': {
                 'bank': '',
                 'branch': '',
                 'type_of_account': '',
                 'account_number': '',
                 'account_name':''
             },
    ...
}
```

```py
# Call to class ElasticSearch
from elastic_flow import MizuhoElastic

mizuho = ElasticSearch()
output = mizuho.process(<json_input>) # NOT path of json file

```
