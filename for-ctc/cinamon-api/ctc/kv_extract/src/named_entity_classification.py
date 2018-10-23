# coding=utf-8
import re
import ngram_models
import math
import regex_named_entities as rne


def rechoice(arr):
    re_str = ""
    for a in arr:
        re_str += "|" + a
    return "(" + re_str[1:] + ")"


def classify_date(text):
    text = re.sub(" ", "", text)
    text = re.sub(u"年Ⅱ月", u"年11月", text)
    text = re.sub(u"年ー月", u"年1月", text)
    if re.search("^" + rne.date_format + "$", text):
        return 1
    if re.search(rne.date_format, text):
        # txt = re.sub(rne.date_range_format, "@date", text)
        return 0.7
    return 0


def classify_age(text):
    if re.search(u"\d+(歳|才)", text) and len(text) < 7:
        return 1
    return 0


def get_money(text):
    return 0


def get_matched_strings(text, regex):
    ans = []
    for match in regex.finditer(text):
        ans.append(match.group(1))
    return ans


def named_entity_recognition(text):
    '''
    Purpose: Get all common tag of named entities such as money, age, date
    so that we can extract value from string containing multiple values
    '''
    # tags = {"Money": [], "Date": [], "Age": []}
    tags = {}
    money_search = re.compile(rne.money_format)
    tags["Money"] = get_matched_strings(text, money_search)
    date_search = re.compile(rne.date_format)
    tags["Date"] = get_matched_strings(text, date_search)
    return tags


def classify_name(text):
    k = min(0, ngram_models.get_prob_name(text.encode("utf8")) / 7.0 + 9)
    k = math.exp(k)
    if k > 0.3 and len(text) > 1 and len(text) < 7:
        if len(text) == 2:
            return 0.7
        else:
            return 1
    return 0


def value_classification(text):
    if isinstance(text, str):
        text = unicode(text)
    text = re.sub("\s+", "", text)
    if re.search("^" + rne.money_format + "$", text):
        return "MONEY", 1
    if re.search("^" + rne.money_range + "$", text):
        return "MONEY", 1
    if re.search("^" + rne.age_format + "$", text):
        return "AGE", 1
    if re.search("^" + rne.payment_method_format + "$", text):
        return "PAYMENT_METHOD", 1
    if re.search("^" + rne.nonfleet_grade_format + "$", text):
        return "NONFLEET_GRADE", 1
    if re.search("^" + rne.yes_no_format + "$", text):
        return "YES_NO", 1
    if re.search("^" + rne.car_model + "$", text):
        return "CAR_MODEL", 1
    if re.search("^" + rne.license_color + "$", text):
        return "LICENSE_COLOR", 1
    if classify_date(text) > 0.6:
        return "DATE", 1
    add_score = 0
    if re.search("\d+", text):
        addresses = re.split("\d+", text)
        for address in addresses:
            t = address
            if len(address) > 5:
                t = address[:3]
            k = min(0, ngram_models.get_prob_location(t) / 5.0 + 8)
            k = math.exp(k)
            if add_score < k:
                add_score = k
    if add_score > 0.6:
        return "ADDRESS", add_score
    if classify_name(text) > 0.5:
        return "NAME", classify_name(text)
    return "OTHER", 0.9
