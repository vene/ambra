'''
Created on Dec 21, 2014

@author: Alina Maria Ciobanu
'''

import numpy
import re

TOKEN_NER_TAGS = ['DATE', 'NUMBER']
WIKI_NER_TAGS = ['PERSON', 'ORGANIZATION', 'LOCATION']

DEFAULT_YEAR_VALUE = 1858 # the mean of the lowest and highest value for all possible intervals, hardcoded for now

def get_temporal_feature(doc):
    flat_tokens = sum(doc['tokens'], [])
    flat_ner = sum(doc['ner'], [])
    zipped = zip(flat_tokens, flat_ner)

    return get_temporal_feature_for_zip(zipped)

def get_temporal_feature_for_zip(zipped):
    """ entry format: [(token_1, ner_tag_1), (token_2, ner_tag_2), (token_3, ner_tag_3)] """
    years = []

    years.extend(get_years_from_token(zipped))
     
    if years:
        return numpy.median(numpy.array(years))
    # TODO check Wiki

    return DEFAULT_YEAR_VALUE

def get_years_from_token(zipped):
    years = []

    for token, ner_tag in zipped:
        if ner_tag in TOKEN_NER_TAGS:
            match = re.match( r'.*(\d{4}).*', token)
            if (match): 
                years.append(int(match.group(1)))

    return years

if __name__ == "__main__":
    zipped = zip(['the 1990s', '1967', '1875', '123x4'], ['DATE', 'DAT', 'NUMBER', 'NUMBER'])
    print get_temporal_feature_for_zip(zipped)    
