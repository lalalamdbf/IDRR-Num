import numpy as np
import pandas as pd
import torch.utils.data as data
import re
import string
import json
from word2number.w2n import word_to_num

NUM_NER_TYPES = ['NUMBER', 'PERCENT','MONEY','TIME','DATE','DURATION','ORDINAL']

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_number_from_word(word, improve_number_extraction=True):
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match('^\d*1st$', word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match('^\d*2nd$', word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match('^\d*3rd$', word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match('^\d+th$', word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if '.' in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                        # per month, e.g "1050.95/month"
                        if '.' in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    return number

def roberta_tokenize(text, tokenizer):
    split_tokens = []

    numbers = []
    number_indices = []
    number_len = []


    prev_is_whitespace = True
    tokens = []
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False
            
    ner_type = []
    for i, token in enumerate(tokens):
        index = token.find("@flag_")
        
        tmp = ""
        if index != -1:
             tmp = token[index+6:]
             token = token[:index]
        if i != 0:
            sub_tokens = tokenizer.tokenize(" " + token)
        else:
            sub_tokens = tokenizer.tokenize(token)
        token_number = get_number_from_word(token)

        if token_number is not None:
            numbers.append(token_number)
            number_indices.append(len(split_tokens))
            number_len.append(len(sub_tokens))
            
            if index != -1:
                ner_type.append(tmp)
                
        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
     

    return split_tokens, numbers, number_indices, number_len, ner_type

def read_data(file_path,tokenizer):
    with open(file_path) as f:
        data = json.load(f)  
        
    for i in range(0,len(data['sentences']),2):
        arg1 = data['sentences'][i]['tokens']
        arg2 = data['sentences'][i+1]['tokens']
        
        arg1_tokens = []
        arg2_tokens = []
        
        for j in range(0,len(arg1)):
            ner_type = ''
            if arg1[j]['ner'] in NUM_NER_TYPES:
                ner_type = '@flag_' + arg1[j]['ner']
            arg1_tokens.append(arg1[j]['originalText'] + ner_type)
            
        for j in range(0,len(arg2)):
            ner_type = ''
            if arg2[j]['ner'] in NUM_NER_TYPES:
                ner_type = '@flag_' + arg2[j]['ner']
            arg2_tokens.append(arg2[j]['originalText'] + ner_type)
        
        arg1 = " ".join(arg1_tokens)
        arg2 = " ".join(arg2_tokens)
        
        arg1_tokens, numbers_in_arg1, arg1_number_indices, arg1_number_len ,arg1_ner_type= roberta_tokenize(
                arg1, tokenizer)
        arg2_tokens, numbers_in_arg2, arg2_number_indices, arg2_number_len ,arg2_ner_type= roberta_tokenize(
                arg2, tokenizer)
        
        return arg1_tokens, numbers_in_arg1, arg1_number_indices, arg1_number_len, arg1_ner_type, \
               arg2_tokens, numbers_in_arg2, arg2_number_indices, arg2_number_len ,arg2_ner_type

