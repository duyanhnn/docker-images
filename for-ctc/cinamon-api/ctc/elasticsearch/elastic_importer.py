import csv
import os
import unicodedata
import json
import pandas as pd 
import numpy as np 
import requests

from operator import itemgetter
from elasticsearch import helpers, Elasticsearch


def preprocess_data(csv_path):
    df = pd.read_csv(csv_path, converters={'bank_code' : str, 'branch_code' : str,'銀行コード':str, '郵便番号':str, 
                                           '地名': str, '口座番号': str, '郵便番号':str})
    df = df.replace('\n','', regex=True)
    df.fillna(axis=1,value='',inplace=True)
    dir_path = os.path.dirname(csv_path)
    filename, _ = os.path.splitext(os.path.basename(csv_path))
    output = os.path.join(dir_path,'data_'+ os.path.basename(csv_path))
    alternatives = {
            '名称 1': 'company_name_1' ,
            '名称 2': 'company_name_2',
            '名称 3': 'company_name_3',
            '名称 4': 'company_name_4', 
            '地名2': 'company_address' ,
            '電話番号' :'tel' ,
            '郵便番号': 'postcode',
            'FAX番号' : 'fax',
            '仕入先ｺｰﾄﾞ': 'company_id',
            '銀行コード': 'bank_branch_code',
            '口座番号': 'account_number',
            '種別': 'type_of_account',
            '口座名義人名': 'account_name'
        }
    for _jp, _en in alternatives.items():
        df.rename(columns = {'{}'.format(_jp):'{}'.format(_en)}, inplace = True)
    for column in df.columns:
        if column in ['tel', 'fax']:
            df[column] = df[column].map(lambda x: ''.join([i for i in str(x) if i.isdigit()]))
            df[column] = df[column].map(lambda x: unicodedata.normalize('NFKC', str(x)))
            df[column] = df[column].map(lambda x:  str(x).replace('',' ').strip())
        elif column in ['bank_code', 'branch_code', 'bank_branch_code', 'company_id', 'postcode']:
            df[column] = df[column].map(lambda x: unicodedata.normalize('NFKC', str(x)))
        else:
            df[column] = df[column].map(lambda x: unicodedata.normalize('NFKC', str(x)))
            df[column] = df[column].map(lambda x:  str(x).replace(' ',''))
            df[column] = df[column].map(lambda x:  str(x).replace('',' ').strip())
    df.to_csv(output, sep=',', encoding='utf-8', index=False)
    return output, filename

class ElasticImporter:
    def __init__(self, csv_path):
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.csv_path = csv_path

    def process(self):
        data_path, index = preprocess_data(self.csv_path)
        with open(data_path) as f:
            reader = csv.DictReader(f)
            helpers.bulk(self.es, reader, index=index, doc_type='mizuho')
        print('Imported the database of {} to Elasticsearch Server!'.format(index))
    
    # def delete_elastic(self):
    #     os.system('curl -XDELETE localhost:9200/mizuho')

def main():
    pass

if __name__ == '__main__':
    main()