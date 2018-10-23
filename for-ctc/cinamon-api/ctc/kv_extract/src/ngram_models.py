# coding=utf-8
import csv
import codecs
import re
import math

START_SYMBOL = '^'
END_SYMBOL = '$'
N = 3
lbda = -3.0
INIT = False
DATA_PATH = "/Users/macbook/GitHub/flax_bprost/debug/"
prefect_ngram = None
ward_ngram = None
street_ngram = None
fname_ngram = None
lname_ngram = None


class Ngram:
    fdict = dict()

    def increase(self, t):
        t = START_SYMBOL + t + END_SYMBOL
        for i in range(len(t) + 1):
            for j in range(max(i - N, 0), i):
                s = t[j:i]
                if s in self.fdict:
                    self.fdict[s] += 1
                else:
                    self.fdict[s] = 1

    def __init__(self, list):
        self.fdict = dict()
        for l in list:
            self.increase(l)
        self.fdict[''] = self.fdict[START_SYMBOL]

    def load_model(self, data_file):
        self.data_file = data_file

    def get_prob3(self, k):
        if k in self.fdict:
            return math.log(1.0 * self.fdict[k] / self.fdict[k[:-1]])
        k = k[1:]
        if len(k) == 0:
            return -1000.0
        if k in self.fdict:
            return math.log(1.0 * self.fdict[k] / self.fdict[k[:-1]]) + lbda
        k = k[1:]
        if len(k) == 0:
            return -1000.0
        if k in self.fdict:
            return math.log(1.0 * self.fdict[k] / self.fdict[START_SYMBOL]) + lbda * 2
        return -1000.0

    def get_prob(self, k):
        ans = 0
        k = START_SYMBOL + k + END_SYMBOL
        for t in range(1, len(k) + 1):
            ant = self.get_prob3(k[max(0, t - N):t])
            ans += self.get_prob3(k[max(0, t - N):t])
        return ans


def get_prob_2names(ngram1, ngram2, s):
    max_prob = -1000
    pos = 0
    for i in range(0, len(s) + 1):
        s1 = s[:i]
        s2 = s[i:]
        prob = ngram1.get_prob(s1) + ngram2.get_prob(s2)
        if prob > max_prob:
            max_prob = prob
            pos = i
    return max_prob, s[:pos], s[pos:]


def get_prob_location(s):
    if len(s) < 2:
        return -1000
    # print(prefect_ngram.get_prob(s), ward_ngram.get_prob(s), street_ngram.get_prob(s))
    return max(prefect_ngram.get_prob(s), ward_ngram.get_prob(s), street_ngram.get_prob(s) * 2)


def get_prob_name(s):
    if len(s) == 4:
        return max(fname_ngram.get_prob(s[:2]), lname_ngram.get_prob(s[2:]), -999)
    return max(get_prob_2names(lname_ngram, fname_ngram, s)[0], lname_ngram.get_prob(s), fname_ngram.get_prob(s))


def init_model(data_path=DATA_PATH):
    global prefect_ngram
    global ward_ngram
    global street_ngram
    global fname_ngram
    global lname_ngram
    dict_prefect = dict()
    dict_prefect_id = dict()
    prefect_num = 0
    dict_ward = dict()
    streets = set()
    with open(data_path + "KEN_ALL_1.csv", "r") as fdata:
        spamreader = csv.reader(fdata, delimiter=',')
        for row in spamreader:
            a, b, c = unicode(row[6], 'utf-8'), unicode(row[7], 'utf-8'), re.sub("（.*?）", "", unicode(row[8], 'utf-8'))
            if a not in dict_prefect_id:
                prefect_num += 1
                dict_prefect_id[a] = prefect_num
                dict_prefect[prefect_num] = a
            if b not in dict_ward:
                dict_ward[b] = [dict_prefect_id[a]]
            else:
                if dict_prefect_id[a] not in dict_ward[b]:
                    dict_ward[b].append(dict_prefect_id[a])
            if c not in streets:
                streets.add(c)
    first_names = []
    with open(data_path + 'name_first_kanji.txt') as f:
        for line in f:
            first_names.append(line)
    fname_ngram = Ngram(first_names)
    last_names = []
    with open(data_path + 'name_last_kanji.txt') as f:
        for line in f:
            last_names.append(line)
    lname_ngram = Ngram(last_names)

    list_s = []
    for key, value in dict_prefect.items():
        list_s.append(value)
    prefect_ngram = Ngram(list_s)
    list_w = []
    for key, value in dict_ward.items():
        list_w.append(key)
    ward_ngram = Ngram(list_w)

    streets = list(streets)
    street_ngram = Ngram(streets)
