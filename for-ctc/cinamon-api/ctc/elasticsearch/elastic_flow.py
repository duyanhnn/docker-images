import csv
import difflib
import os
import unicodedata
import json
import pandas as pd 
import numpy as np 
import requests
import re

from operator import itemgetter
from elasticsearch import helpers, Elasticsearch
# import logging

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     filename='/Users/duongthanh/Documents/Mizuho_2/myapp.log',
#                     filemode='w')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logger = logging.getLogger('')
# logger.addHandler(console)

# logging.info("MIZUHO'S PROJECT")
# logging.getLogger('elasticsearch').setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logger1 = logging.getLogger('elastic')
# logger1.setLevel(logging.DEBUG)


def load_json(json_path):
    with open(json_path, 'r') as json_data:
        ocr_result = json.load(json_data)
    return ocr_result

class MizuhoElastic:
    def __init__(self, debug = False):
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.company_result = {}
        self.bank_result = {}
        self.company_json = {}
        self.bank_json = {}
        self.company_id = {}
        self.debug = debug
        self.indexs = {
            'company_information': ['company_name', 'company_address', 'tel', 'fax', 'postcode'],
            'bank_information': ['bank']
        }
    
    def find_index(self, field):
        index_out = ''
        for index, fields in self.indexs.items():
            if field in fields:
                index_out = index
                break
        return index_out
    
    def normalize_output(self, old_string):
        new_string = unicodedata.normalize('NFKC', old_string.replace(' ', '').strip())
        return new_string
    
    def normalize_input(self, old_string):
        new_string = unicodedata.normalize('NFKC', str(old_string).replace(' ', ''))
        new_string = unicodedata.normalize('NFKC', new_string.replace('', ' ').strip())
        return new_string
    
    def formated(self, field, value):
        if value:
            if field.startswith('tel'):
                value = ''.join([i for i in str(value) if i.isdigit()])
                value = self.normalize_input(value)
            elif field.startswith('company_address'):
                value = value.replace('東京都','')
                value = value.replace('丁目','-').replace('番','-').replace('号', '')
                value = self.normalize_input(value)
            elif field.startswith('bank_branch_code'):
                pass
            else:
                value = self.normalize_input(value)
            return value
        else:
            return ''

    def search_a_field(self, index, type_field, value_field):
        # if self.debug:
        #     logger1.debug('> {query: {index: %s, type: %s, value: %s}}' %(index, type_field, value_field if value_field else 'NONE' ))
        _ids = {}
        # if value_field:
        try:
            value_field = self.formated(type_field,value_field)
            body={"size":100, "query": {"match" : { "{}".format(type_field): { 
                                                            "cutoff_frequency" : 0.1, 
                                                            "query": "{}".format(value_field) } }}}

            res = self.es.search(index=index, doc_type='mizuho', body=body)
            print(res)
            res = res['hits']['hits']
            for result in res:
                _score = 0.0
                value_search =  self.normalize_output(result['_source']['{}'.format(type_field)])
                diff = difflib.SequenceMatcher(None, self.normalize_output(value_field), value_search)
                _score = diff.ratio()
                if _score > 0.7:
                    _ids[result['_source']['company_id']] = _score
                        # if self.debug:
                        #     logger1.debug('< {result: {company_id: %s, result: %s, score: %s}}' 
                        #                   %(result['_source']['company_id'], value_search, round(_score,2)))
        except AttributeError:
            print('Error with int object')
        return _ids
    
    def search_multiple_fields(self, index, type_field, value_field, ratio =0.7):
        # if self.debug:
        #     logger1.debug('> {query: {index: %s, type: %s, value: %s}}' %(index, type_field, value_field if value_field else 'NONE' ))
        _ids = {}
        value_field = self.normalize_input(value_field)
        body =  {"size":500, "query": {"multi_match":{
            "fields": ["{}*".format(type_field)],
            "query" : "{}".format(value_field),
            "type": "most_fields"
        }}}
        res = self.es.search(index=index, doc_type='mizuho', body=body)
        res = res['hits']['hits']
        _ids = {}
        try:
            for result in res:
                _score = 0.0
                for _field, _value in result['_source'].items():
                    if _field.startswith('{}'.format(type_field)):
                        value_search =  self.normalize_output(_value)
                        diff = difflib.SequenceMatcher(None, self.normalize_output(value_field), value_search)
                        if diff.ratio() > ratio and diff.ratio() > _score:
                            _score = diff.ratio()
                            _ids[result['_source']['company_id']] = _score
        except:
            pass
        return _ids

    def search_two_fields(self, index, type_field1, value_field1, type_field2, value_field2):
        # if self.debug:
        #     logger1.debug('> {query: {index: %s, type: %s, value: %s}}' %(index, type_field, value_field if value_field else 'NONE' ))
        _ids = {}
        body = {"query": {"bool": {"should": [
                                        {
                    "match": {
                        type_field1: {
                            "cutoff_frequency": 0.1,
                            "query": value_field1 
                        }
                    }
                },
                 {
                    "match": {
                        type_field2: {
                            "cutoff_frequency": 0.1,
                            "query": value_field2
                        }
                    }
                } 
                                            ]}}}
        res = self.es.search(index=index, doc_type='mizuho', body=body)
        res = res['hits']['hits']
        _ids = {}
        try:
            for result in res:
                _score = 0.0
        except:
            pass
        return _ids

    def search_company_fields(self):
        if any([v for k, v in self.company_result.items()]):
            _ids = self.company_true_id()
            if _ids is not None:
                _id = _ids[0]
                self.company_id = _id
                body = {"query": {"term": {"company_id": "{}".format(self.company_id)}}}
                result_from_id = self.es.search(index='company_information', doc_type='mizuho', body= body)
                result_from_id = result_from_id['hits']['hits'][0]
                if result_from_id.get('_source'):
                    _score_company_name = 0
                    x = list(self.company_result.copy())
                    for field in x:
                        value_field = self.company_result.get(field)
                        if field.startswith('company_name') and value_field:
                            fields = [i for i in result_from_id['_source'].keys() if i.startswith('company_name')]
                            for field_result in fields:
                                _diff = difflib.SequenceMatcher(None, self.normalize_output(result_from_id['_source'][field_result]), 
                                                            self.company_json['company_name'])
                                _score = _diff.ratio( ) 
                                if _score > _score_company_name:
                                    self.company_result['company_name'] = self.normalize_output(result_from_id['_source'][field_result])
                                    _score_company_name = _score
                            continue
                        elif field.startswith('company_name') and not value_field:
                            self.company_result['company_name'] = self.normalize_output(result_from_id['_source']['company_name_1'])
                            continue
                        elif field in self.indexs['company_information']:
                            self.company_result[field] =  self.normalize_output(result_from_id['_source'][field])
                        else:
                            self.company_result.pop(field, None)
        return self.company_result
            
    def company_true_id(self):
        _total_ids = []
        _ids = set([i for k,v in self.company_result.items() for i in v.keys()])
        for _id in _ids:
            _count = sum([1 for k,v in self.company_result.items() for i in v.keys() if i == _id])
            _score = sum([v[i] for k,v in self.company_result.items() for i in v.keys() if i == _id])
            _total_ids.append((_id, _count, _score))    # (id, _count, _total_score)
        _total_ids = sorted(_total_ids, key=itemgetter(1,2), reverse=True)    # sorting _count to _total_score
        if len(_total_ids) > 0:
            return _total_ids[0]
        else:
            return None
    
    def lookup_bankcode(self, bank_code, branch_code):
        body = {"query": {"bool": {"must": [
                                        {"term": { "bank_code": "{}".format(bank_code)}},
                                        {"term": { "branch_code": "{}".format(branch_code)}}
                                            ]}}}
        result_bank = self.es.search(index='bank_information', doc_type='mizuho', body= body)
        bank_name = self.normalize_output(result_bank['hits']['hits'][0]['_source']['bank_name'])
        branch_name = self.normalize_output(result_bank['hits']['hits'][0]['_source']['branch_name'])
        print(bank_name,branch_name)
        return bank_name, branch_name
    
    def lookup_bankname(self, bank_name, branch_name):
        u_range = {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")}
        is_kata_bank = any([u_range["from"] <= ord(c) <= u_range["to"]] for c in bank_name)
        is_kata_branch = any([u_range["from"] <= ord(c) <= u_range["to"]] for c in branch_name)
        if is_kata_bank and bank_name.endswith('銀行'):
            bank_name = bank_name.replace('銀行', '')
        if is_kata_branch and branch_name.endswith('支店'):
            branch_name = branch_name.replace('支店', '')
        bank_name= self.normalize_input(bank_name)
        branch_name= self.normalize_input(branch_name)
        body = {"size": 5,"query": {"bool": {
            "should": [
                {
                    "match": {
                        "bank_name": {
                            "cutoff_frequency": 0.1,
                            "query": bank_name
                        }
                    }
                },
                 {
                    "match": {
                        "bank_name_kana": {
                            "cutoff_frequency": 0.1,
                            "query": bank_name
                        }
                    }
                },
                {
                    "match": {
                        "branch_name": {
                            "cutoff_frequency": 0.1,
                            "query": branch_name
                        }
                    }
                },
                {
                    "match": {
                        "branch_name_kana": {
                            "cutoff_frequency": 0.1,
                            "query": branch_name
                        }
                    }
                }
                ]}}}
        result_bank = self.es.search(index='bank_information', doc_type='mizuho', body= body)
        result_bank = result_bank['hits']['hits']
        _result = {}
        for result in result_bank:
            bank_name_db = self.normalize_output(result['_source']['bank_name'])
            bank_name_kana_db = self.normalize_output(result['_source']['bank_name_kana'])
            branch_name_db = self.normalize_output(result['_source']['branch_name'])
            branch_name_kana_db = self.normalize_output(result['_source']['branch_name_kana'])
            bank_code = self.normalize_output(result['_source']['bank_code'])
            branch_code = self.normalize_output(result['_source']['branch_code'])
            diff_bank = difflib.SequenceMatcher(None,self.normalize_output(bank_name), bank_name_db)
            diff_bank_kana = difflib.SequenceMatcher(None,self.normalize_output(bank_name), bank_name_kana_db)
            diff_branch = difflib.SequenceMatcher(None,self.normalize_output(branch_name), branch_name_db)
            diff_branch_kana = difflib.SequenceMatcher(None,self.normalize_output(branch_name), branch_name_kana_db)
            bank_score = max([diff_bank.ratio(),diff_bank_kana.ratio()])
            branch_score = max([diff_branch.ratio(),diff_branch_kana.ratio()])
            if bank_score + branch_score >= 1.5:
                _result[bank_code+branch_code] = branch_score + bank_score
        try:
            bank_branch_code = max(_result.items(), key=itemgetter(1))[0]
        except ValueError:
            bank_branch_code = ''
            pass
        return bank_branch_code
    
    def search_bank_fields(self):
        if self.company_id:
            body = {"query": {"term": {"company_id": "{}".format(self.company_id)}}}
            result_from_id = self.es.search(index='company_information', doc_type='mizuho', body= body)
            result_from_id = result_from_id['hits']['hits']
            if result_from_id:
                for _index, result in enumerate(result_from_id):
                    bank_code = self.normalize_output(result['_source']['bank_branch_code'])[:4]
                    branch_code = self.normalize_output(result['_source']['bank_branch_code'])[4:]
                    bank_name, branch_name = self.lookup_bankcode(bank_code, branch_code)
                    self.bank_result['bank_database_'+ str(_index+1)] = {
                        'bank': bank_name ,
                        'branch': branch_name,
                        'type_of_account': self.normalize_output(result['_source']['type_of_account']),
                        'account_name': self.normalize_output(result['_source']['account_name']),
                        'account_number': self.normalize_output(result['_source']['account_number']),
                    }
        pass

    def process(self,json_input):
        self.company_json = {}
        self.bank_json = {}
        bank_index = 1
        for field, value_field in json_input.items():
            index = self.find_index(field)
            if not value_field:
                self.company_result[field] = {}
                continue
            if index.startswith('company_information'):
                self.company_json[field] = value_field
                if field.startswith('company_name'):
                    self.company_result[field] = self.search_multiple_fields(index, field ,value_field)
                    continue
                self.company_result[field] = self.search_a_field(index, field, value_field)
            else:
                self.company_result['account_name'+str(bank_index)] = self.search_a_field('company_information','account_name', 
                                                                                              value_field['account_name'])
                self.company_result['account_number'+str(bank_index)] = self.search_a_field('company_information','account_number', 
                                                                                              value_field['account_number'])
                bank_index += 1
        print(self.company_result)
        self.search_company_fields()
        self.search_bank_fields()
        json_result_all = {**self.company_result, **self.bank_result}
        return json_result_all

def main():
    #SAMPLE TO RUN
    
    pass

if __name__ == '__main__':
    main()