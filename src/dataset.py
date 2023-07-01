import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import re
import math
import string
from collections import defaultdict, OrderedDict
from itertools import chain
from tqdm import tqdm
from util import *
from word2number.w2n import word_to_num


def cached_path(file_path):
    return file_path


NUM_NER_TYPES = ['NUMBER', 'PERCENT', 'MONEY',
    'TIME', 'DATE', 'DURATION', 'ORDINAL']

ner_map = defaultdict(lambda: -1, {
        "NUMBER": 1,
        "PERCENT": 2,
        "MONEY": 3,
        "TIME": 4,
        "DATE": 5,
        "DURATION": 6,
        "ORDINAL": 7})

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
                ner_type.append(ner_map[tmp])
            else:
                ner_type.append(1)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)

    return split_tokens, numbers, number_indices, number_len, ner_type


def read_data(file_path, tokenizer):
    with open(file_path) as f:
        data = json.load(f)

    for i in range(0, len(data['sentences']), 2):
        arg1 = data['sentences'][i]['tokens']
        arg2 = data['sentences'][i+1]['tokens']

        arg1_tokens = []
        arg2_tokens = []

        for j in range(0, len(arg1)):
            ner_type = ''
            if arg1[j]['ner'] in NUM_NER_TYPES:
                ner_type = '@flag_' + arg1[j]['ner']
            arg1_tokens.append(arg1[j]['originalText'] + ner_type)

        for j in range(0, len(arg2)):
            ner_type = ''
            if arg2[j]['ner'] in NUM_NER_TYPES:
                ner_type = '@flag_' + arg2[j]['ner']
            arg2_tokens.append(arg2[j]['originalText'] + ner_type)

        arg1 = " ".join(arg1_tokens)
        arg2 = " ".join(arg2_tokens)

        arg1_tokens, numbers_in_arg1, arg1_number_indices, arg1_number_len, arg1_ner_type = roberta_tokenize(
            arg1, tokenizer)
        arg2_tokens, numbers_in_arg2, arg2_number_indices, arg2_number_len, arg2_ner_type = roberta_tokenize(
            arg2, tokenizer)

        return arg1_tokens, numbers_in_arg1, arg1_number_indices, arg1_number_len, arg1_ner_type, \
            arg2_tokens, numbers_in_arg2, arg2_number_indices, arg2_number_len, arg2_ner_type


class Dataset(data.Dataset):
    rel_map_14 = defaultdict(lambda: -1, {
        "Comparison.Concession": 0,
        "Comparison.Contrast": 1,

        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 3,
        "Contingency.Condition": 4,

        "Expansion.Alternative": 5,
        "Expansion.Alternative.Chosen alternative": 6,
        "Expansion.Conjunction": 7,
        "Expansion.Exception": 8,
        "Expansion.Instantiation": 9,
        "Expansion.Restatement": 10,

        # "Temporal",
        "Temporal.Asynchronous.Precedence": 11,
        "Temporal.Asynchronous.Succession": 12,
        "Temporal.Synchrony": 13})
    # Counter({
    #     "Comparison.Concession": 206,
    #     "Comparison.Contrast": 1872,
    #     "Contingency.Cause.Reason": 2287,
    #     "Contingency.Cause.Result": 1530,
    #     "Contingency.Condition": 2,
    #     "Expansion.Alternative": 12,
    #     "Expansion.Alternative.Chosen alternative": 159,
    #     "Expansion.Conjunction": 3577,
    #     "Expansion.Exception": 1,
    #     "Expansion.Instantiation": 1251,
    #     "Expansion.Restatement": 2807,
    #     "Temporal.Asynchronous.Precedence": 467,
    #     "Temporal.Asynchronous.Succession": 133,
    #     "Temporal.Synchrony": 236})

    rel_map_11 = defaultdict(lambda: -1, {
        # "Comparison",
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 1,
        "Comparison.Contrast.Juxtaposition": 1,
        "Comparison.Contrast.Opposition": 1,
        # "Comparison.Pragmatic concession",
        # "Comparison.Pragmatic contrast",

        # "Contingency",
        "Contingency.Cause": 2,
        "Contingency.Cause.Reason": 2,
        "Contingency.Cause.Result": 2,
        "Contingency.Pragmatic cause.Justification": 3,
        # "Contingency.Condition",
        # "Contingency.Condition.Hypothetical",
        # "Contingency.Pragmatic condition.Relevance",

        # "Expansion",
        "Expansion.Alternative": 4,
        "Expansion.Alternative.Chosen alternative": 4,
        "Expansion.Alternative.Conjunctive": 4,
        "Expansion.Conjunction": 5,
        "Expansion.Instantiation": 6,
        "Expansion.List": 7,
        "Expansion.Restatement": 8,
        "Expansion.Restatement.Equivalence": 8,
        "Expansion.Restatement.Generalization": 8,
        "Expansion.Restatement.Specification": 8,
        # "Expansion.Alternative.Disjunctive",
        # "Expansion.Exception",

        # "Temporal",
        "Temporal.Asynchronous.Precedence": 9,
        "Temporal.Asynchronous.Succession": 9,
        "Temporal.Synchrony": 10})
    # Counter({
    #      "Comparison.Concession": 216,
    #      "Comparison.Contrast": 1915,
    #      "Contingency.Cause": 3833,
    #      "Contingency.Pragmatic cause": 78,
    #      "Expansion.Alternative": 171,
    #      "Expansion.Conjunction": 3355,
    #      "Expansion.Instantiation": 1332,
    #      "Expansion.List": 360,
    #      "Expansion.Restatement": 2945,
    #      "Temporal.Asynchronous": 662,
    #      "Temporal.Synchrony": 245})

    rel_map_4 = defaultdict(lambda: -1, {
        "Comparison": 0,
        "Comparison.Concession": 0,
        "Comparison.Concession.Contra-expectation": 0,
        "Comparison.Concession.Expectation": 0,
        "Comparison.Contrast": 0,
        "Comparison.Contrast.Juxtaposition": 0,
        "Comparison.Contrast.Opposition": 0,
        "Comparison.Pragmatic concession": 0,
        "Comparison.Pragmatic contrast": 0,

        "Contingency": 1,
        "Contingency.Cause": 1,
        "Contingency.Cause.Reason": 1,
        "Contingency.Cause.Result": 1,
        "Contingency.Condition": 1,
        "Contingency.Condition.Hypothetical": 1,
        "Contingency.Pragmatic cause.Justification": 1,
        "Contingency.Pragmatic condition.Relevance": 1,

        "Expansion": 2,
        "Expansion.Alternative": 2,
        "Expansion.Alternative.Chosen alternative": 2,
        "Expansion.Alternative.Conjunctive": 2,
        "Expansion.Alternative.Disjunctive": 2,
        "Expansion.Conjunction": 2,
        "Expansion.Exception": 2,
        "Expansion.Instantiation": 2,
        "Expansion.List": 2,
        "Expansion.Restatement": 2,
        "Expansion.Restatement.Equivalence": 2,
        "Expansion.Restatement.Generalization": 2,
        "Expansion.Restatement.Specification": 2,

        "Temporal": 3,
        "Temporal.Asynchronous.Precedence": 3,
        "Temporal.Asynchronous.Succession": 3,
        "Temporal.Synchrony": 3})
    # Counter({
    #     "Comparison": 2293,
    #     "Contingency": 3917,
    #     "Expansion": 8256,
    #     "Temporal": 909})

    def __init__(self, data=None, tokenizer=None):
        super(Dataset, self).__init__()
        if data is not None:
            self.data = data
        else:
            self.data = list()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def batchify_no_numerical(batch, rel_map, min_arg=3, max_arg=512, pad_id=0):
        assert isinstance(rel_map, defaultdict)
        len_rel = max(rel_map.values())+1
        arg_size = list()
        for x in batch:
            arg_size = list(x["arg1"].size())[1:]
            break

        valid_batch, prefered_relation = list(), list()
        for i, x in enumerate(batch):
            for r in x["rel_strs"]:
                idx = rel_map[r]
                if idx != -1:
                    valid_batch.append(x)
                    prefered_relation.append(idx)
                    break
        bsz = len(valid_batch)

        arg1_lens = [x["arg1"].size(0) for x in valid_batch]
        arg2_lens = [x["arg2"].size(0) for x in valid_batch]
        max_arg1 = min(max(min_arg, max(arg1_lens)), max_arg)
        max_arg2 = min(max(min_arg, max(arg2_lens)), max_arg)
        arg1 = torch.empty([bsz, max_arg1] + arg_size,
                           dtype=torch.long).fill_(pad_id)
        arg1_mask = batch_convert_len_to_mask(arg1_lens, max_arg1)
        arg2 = torch.empty([bsz, max_arg2] + arg_size,
                           dtype=torch.long).fill_(pad_id)
        arg2_mask = batch_convert_len_to_mask(arg2_lens, max_arg2)

        _id, relation = list(), list()
        for i, x in enumerate(valid_batch):
            _id.append(x["id"])
            l1 = arg1_lens[i]
            if l1 < max_arg1:
                arg1[i, :l1].data.copy_(x["arg1"])
            elif l1 == max_arg1:
                arg1[i].data.copy_(x["arg1"])
            else:
                arg1[i].data.copy_(x["arg1"][:max_arg1])
            l2 = arg2_lens[i]
            if l2 < max_arg2:
                arg2[i, :l2].data.copy_(x["arg2"])
            elif l2 == max_arg2:
                arg2[i].data.copy_(x["arg2"])
            else:
                arg2[i].data.copy_(x["arg2"][:max_arg2])
            rel = torch.zeros((len_rel,), dtype=torch.float,
                              requires_grad=False)
            for r in x["rel_strs"]:
                idx = rel_map[r]
                if idx != -1:
                    rel[idx] = 1
            relation.append(rel)
        relation = torch.cat(relation, dim=0).view(bsz, len_rel)
        prefered_relation = torch.tensor(prefered_relation, dtype=torch.long)

        return _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation

    def batchify_numerical(batch, rel_map, min_arg=3, max_arg=512, pad_id=0):
        assert isinstance(rel_map, defaultdict)
        len_rel = max(rel_map.values())+1
        arg_size = list()
        for x in batch:
            arg_size = list(x["arg1"].size())[1:]
            break

        valid_batch, prefered_relation = list(), list()
        for i, x in enumerate(batch):
            for r in x["rel_strs"]:
                idx = rel_map[r]
                if idx != -1:
                    valid_batch.append(x)
                    prefered_relation.append(idx)
                    break
        bsz = len(valid_batch)

        arg1_lens = [x["arg1"].size(0) for x in valid_batch]
        arg2_lens = [x["arg2"].size(0) for x in valid_batch]
        max_arg1 = min(max(min_arg, max(arg1_lens)), max_arg)
        max_arg2 = min(max(min_arg, max(arg2_lens)), max_arg)
        arg1 = torch.empty([bsz, max_arg1] + arg_size,
                           dtype=torch.long).fill_(pad_id)
        arg1_mask = batch_convert_len_to_mask(arg1_lens, max_arg1)
        arg2 = torch.empty([bsz, max_arg2] + arg_size,
                           dtype=torch.long).fill_(pad_id)
        arg2_mask = batch_convert_len_to_mask(arg2_lens, max_arg2)

        arg1_number_indices_lens = [
            len(x["arg1_number_indices"]) for x in valid_batch]
        arg2_number_indices_lens = [
            len(x["arg2_number_indices"]) for x in valid_batch]
        max_arg1_number_indices_len = max(
            [1] + [x for x in arg1_number_indices_lens])
        max_arg2_number_indices_len = max(
            [1] + [x for x in arg2_number_indices_lens])
        arg1_number_indices = torch.empty(
            [bsz, max_arg1_number_indices_len], dtype=torch.long).fill_(-1)
        arg2_number_indices = torch.empty(
            [bsz, max_arg2_number_indices_len], dtype=torch.long).fill_(-1)
        arg1_number_order = torch.empty(
            [bsz, max_arg1_number_indices_len], dtype=torch.long).fill_(-1)
        arg2_number_order = torch.empty(
            [bsz, max_arg2_number_indices_len], dtype=torch.long).fill_(-1)
        arg1_ner_type = torch.empty(
            [bsz, max_arg1_number_indices_len], dtype=torch.long).fill_(-1)
        arg2_ner_type = torch.empty(
            [bsz, max_arg2_number_indices_len], dtype=torch.long).fill_(-1)

        _id, relation = list(), list()
        for i, x in enumerate(valid_batch):
            _id.append(x["id"])
            l1 = arg1_lens[i]
            if l1 < max_arg1:
                arg1[i, :l1].data.copy_(x["arg1"])
            elif l1 == max_arg1:
                arg1[i].data.copy_(x["arg1"])
            else:
                arg1[i].data.copy_(x["arg1"][:max_arg1])
            arg1_number_size = len(x["arg1_number_indices"]) - 1
            if(arg1_number_size > 0):
                arg1_number_indices[i][:arg1_number_size] = torch.tensor(
                    x["arg1_number_indices"][:arg1_number_size], dtype=torch.long)
                arg1_number_order[i][:arg1_number_size] = torch.tensor(
                    x["arg1_number_order"][:arg1_number_size], dtype=torch.long)
                arg1_ner_type[i][:arg1_number_size] = torch.tensor(
                    x["arg1_ner_type"][:arg1_number_size], dtype=torch.long)

            l2 = arg2_lens[i]
            if l2 < max_arg2:
                arg2[i, :l2].data.copy_(x["arg2"])
            elif l2 == max_arg2:
                arg2[i].data.copy_(x["arg2"])
            else:
                arg2[i].data.copy_(x["arg2"][:max_arg2])

            arg2_number_size = len(x["arg2_number_indices"]) - 1
            if(arg2_number_size > 0):
                arg2_number_indices[i][:arg2_number_size] = l1 + torch.tensor(
                    x["arg2_number_indices"][:arg2_number_size], dtype=torch.long)
                arg2_number_order[i][:arg2_number_size] = torch.tensor(
                    x["arg2_number_order"][:arg2_number_size], dtype=torch.long)
                arg2_ner_type[i][:arg2_number_size] = torch.tensor(
                    x["arg2_ner_type"][:arg2_number_size], dtype=torch.long)

            rel = torch.zeros((len_rel,), dtype=torch.float,
                              requires_grad=False)
            for r in x["rel_strs"]:
                idx = rel_map[r]
                if idx != -1:
                    rel[idx] = 1
            relation.append(rel)
        relation = torch.cat(relation, dim=0).view(bsz, len_rel)
        prefered_relation = torch.tensor(prefered_relation, dtype=torch.long)

        return _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation, arg1_number_indices, arg1_number_order, arg1_ner_type, arg2_number_indices, arg2_number_order, arg2_ner_type

    def get_numerical_dataset(self, original_file_path, json_ner_file_path):
        self.data = list()

        with open(original_file_path) as f1:
            original_data = json.load(f1)
            
        with open(json_ner_file_path) as f2:
            ner_data = json.load(f2)
            
        for i in range(0, len(ner_data['sentences']), 2):
            id = original_data[int(i/2)]['id']
            rel_strs = original_data[int(i/2)]['rel_strs']
            arg1 = ner_data['sentences'][i]['tokens']
            arg2 = ner_data['sentences'][i+1]['tokens']
            
            arg1_tokens = []
            arg2_tokens = []

            for j in range(0, len(arg1)):
                ner_type = ''
                if arg1[j]['ner'] in NUM_NER_TYPES:
                    ner_type = '@flag_' + arg1[j]['ner']
                arg1_tokens.append(arg1[j]['originalText'] + ner_type)

            for j in range(0, len(arg2)):
                ner_type = ''
                if arg2[j]['ner'] in NUM_NER_TYPES:
                    ner_type = '@flag_' + arg2[j]['ner']
                arg2_tokens.append(arg2[j]['originalText'] + ner_type)

            arg1 = " ".join(arg1_tokens)
            arg2 = " ".join(arg2_tokens)

            arg1_tokens, numbers_in_arg1, arg1_number_indices, arg1_number_len, arg1_ner_type = roberta_tokenize(
                arg1, self.tokenizer)
            arg2_tokens, numbers_in_arg2, arg2_number_indices, arg2_number_len, arg2_ner_type = roberta_tokenize(
                arg2, self.tokenizer)


            arg1_tokens.insert(0, self.tokenizer.cls_token)
            arg1_tokens.append(self.tokenizer.sep_token)
            arg2_tokens.insert(0,self.tokenizer.sep_token)
            arg2_tokens.append(self.tokenizer.sep_token)

            def get_number_order(numbers):
                if len(numbers) < 1:
                    return None
                ordered_idx_list = np.argsort(np.array(numbers)).tolist()

                rank = 0
                number_rank = []
                for i, idx in enumerate(ordered_idx_list):
                    if i == 0 or numbers[ordered_idx_list[i]] != numbers[ordered_idx_list[i - 1]]:
                        rank += 1
                    number_rank.append(rank)

                ordered_idx_rank = zip(ordered_idx_list, number_rank)

                final_rank = sorted(ordered_idx_rank, key=lambda x: x[0])
                final_rank = [item[1] for item in final_rank]

                return final_rank

            all_number = numbers_in_arg1 + numbers_in_arg2
            all_number_order = get_number_order(all_number)

            if all_number_order is None:
                arg1_number_order = []
                arg2_number_order = []
            else:
                arg1_number_order = all_number_order[:len(numbers_in_arg1)]
                arg2_number_order = all_number_order[len(numbers_in_arg1):]

            arg1_number_indices = [
                indice + 1 for indice in arg1_number_indices]
            arg2_number_indices = [
                indice + 1 for indice in arg2_number_indices]

            # hack to guarantee minimal length of padded number
            arg1_number_indices.append(-1)
            arg1_number_order.append(-1)

            arg2_number_indices.append(-1)
            arg2_number_order.append(-1)
            
            arg1_ner_type.append(-1)
            arg2_ner_type.append(-1)
            
            arg1_number_len.append(-1)
            arg2_number_len.append(-1)

            arg1_number_order = np.array(arg1_number_order)
            arg2_number_order = np.array(arg2_number_order)

            x = {"id": id,
                 "rel_strs": rel_strs,
                 "arg1": torch.tensor(self.tokenizer.convert_tokens_to_ids(arg1_tokens)),
                 "arg1_token": arg1_tokens,
                 "arg1_number_indices": arg1_number_indices,
                 "arg1_number_order": arg1_number_order,
                 "arg1_number_len": arg1_number_len,
                 "arg1_ner_type": arg1_ner_type,
                 "arg2": torch.tensor(self.tokenizer.convert_tokens_to_ids(arg2_tokens)),
                 "arg2_token": arg2_tokens,
                 "arg2_number_indices": arg2_number_indices,
                 "arg2_number_order": arg2_number_order,
                 "arg2_number_len": arg2_number_len,
                 "arg2_ner_type": arg2_ner_type}
            # if len(arg1_number_indices) + len(arg2_number_indices) > 4:
            self.data.append(x)

    def get_dataset_pdtb(self, csv_file_path, sections=None, types=None, is_tokenizer=False):
        self.data = list()

        df = pd.read_csv(csv_file_path, usecols=[
            "Relation", "Section", "FileNumber", "SentenceNumber",
            "ConnHeadSemClass1", "ConnHeadSemClass2",
            "Conn2SemClass1", "Conn2SemClass2",
            "Arg1_RawText", "Arg2_RawText", "FullRawText"])
        if sections:
            df = df[df["Section"].isin(set(sections))]
        if types:
            df = df[df["Relation"].isin(set(types))]
        df.fillna("", inplace=True)

        parsed_result = list()
        for idx, row in tqdm(df.iterrows()):
            rel_strs = list()
            if row[0] == "EntRel":
                rel_strs.append(row[0])
            else:
                if row[4]:
                    rel_strs.append(row[4])
                if row[5]:
                    rel_strs.append(row[5])
                if row[6]:
                    rel_strs.append(row[6])
                if row[7]:
                    rel_strs.append(row[7])
                    
            arg1 = self.tokenizer.tokenize(row[8])
            arg2 = self.tokenizer.tokenize(row[9])
            arg1.insert(0, self.tokenizer.cls_token)
            arg1.append(self.tokenizer.sep_token)
            arg2.insert(0, self.tokenizer.sep_token)
            arg2.append(self.tokenizer.sep_token)
            
            x = {"id": "%d_%s_wsj_%02d%02d" % (idx, row[0], row[1], row[2]),
                 "rel_strs": rel_strs,
                 "arg1": torch.tensor(self.tokenizer.convert_tokens_to_ids(arg1)) if(is_tokenizer) else row[8],
                 "arg2": torch.tensor(self.tokenizer.convert_tokens_to_ids(arg2)) if(is_tokenizer) else row[9]}
            self.data.append(x)
            
    def get_txt_pdtb(self, csv_file_path, dataset_file_path, sections=None, types=None, is_tokenizer=False):
        self.data = list()

        df = pd.read_csv(csv_file_path, usecols=[
            "Relation", "Section", "FileNumber", "SentenceNumber",
            "ConnHeadSemClass1", "ConnHeadSemClass2",
            "Conn2SemClass1", "Conn2SemClass2",
            "Arg1_RawText", "Arg2_RawText", "FullRawText"])
        if sections:
            df = df[df["Section"].isin(set(sections))]
        if types:
            df = df[df["Relation"].isin(set(types))]
        df.fillna("", inplace=True)

        with open(dataset_file_path, 'w') as f:
            for idx, row in tqdm(df.iterrows()):                     
                arg1 = row[8]
                arg2 = row[9]
                f.write(arg1 + '\n' + arg2 + '\n')
        f.close()
            
    def get_dataset_conll(self, json_file_path, types, is_tokenizer=False):
        self.data = list()

        df = pd.read_json(json_file_path, lines=True)

        if types:
            df = df[df["Type"].isin(set(types))]
        df.fillna("", inplace=True)

        for idx, row in tqdm(df.iterrows()):
            rel_strs = row[5]
        
        
            arg1 = self.tokenizer.tokenize(row[0]["RawText"])
            arg2 = self.tokenizer.tokenize(row[1]["RawText"])
            arg1.insert(0, self.tokenizer.cls_token)
            arg1.append(self.tokenizer.sep_token)
            arg2.insert(0, self.tokenizer.sep_token)
            arg2.append(self.tokenizer.sep_token)
            x = {"id": "%d_%s_%s" % (idx, row[6], row[3]),
                "rel_strs": rel_strs,
                "arg1": torch.tensor(self.tokenizer.convert_tokens_to_ids(arg1)) if(is_tokenizer) else row[0]["RawText"],
                "arg2": torch.tensor(self.tokenizer.convert_tokens_to_ids(arg2)) if(is_tokenizer) else row[1]["RawText"]}
            self.data.append(x)
    
    def get_txt_conll(self, json_file_path, dataset_file_path, types, is_tokenizer=False):
        self.data = list()

        df = pd.read_json(json_file_path, lines=True)

        if types:
            df = df[df["Type"].isin(set(types))]
        df.fillna("", inplace=True)

        with open(dataset_file_path, 'w') as f:
            for idx, row in tqdm(df.iterrows()):                     
                arg1 = row[0]
                arg2 = row[1]
                f.write(arg1 + '\n' + arg2 + '\n')
        f.close()
            
    def load_pt(self, pt_file_path):
        self.data = torch.load(pt_file_path)
        return self

    def save_pt(self, pt_file_path):
        torch.save(self.data, pt_file_path,
                   _use_new_zipfile_serialization=False)
        
    def save_json(self, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8', newline='') as out_file:
            json.dump(self.data, out_file, ensure_ascii=False)


class Sampler(data.Sampler):

    def __init__(self, dataset, group_by, batch_size, shuffle=False, drop_last=False):
        super(Sampler, self).__init__(dataset)
        if isinstance(group_by, str):
            group_by = [group_by]
        self.group_by = group_by
        self.cache = OrderedDict()
        for attr in group_by:
            self.cache[attr] = np.array([x[attr]
                                        for x in dataset], dtype=np.float32)
        self.data_size = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def make_array(self):
        rand = np.random.rand(self.data_size).astype(np.float32)
        array = np.stack(list(self.cache.values()) + [rand], axis=-1)
        array = array.view(list(zip(list(self.cache.keys()) +
                           ["rand"], [np.float32] * (len(self.cache) + 1)))).flatten()

        return array

    def handle_singleton(self, batches):
        if not self.drop_last and len(batches) > 1 and len(batches[-1]) < self.batch_size // 2:
            merged_batch = np.concatenate([batches[-2], batches[-1]], axis=0)
            batches.pop()
            batches.pop()
            batches.append(merged_batch[:len(merged_batch)//2])
            batches.append(merged_batch[len(merged_batch)//2:])

        return batches

    def __iter__(self):
        array = self.make_array()
        indices = np.argsort(array, axis=0, order=self.group_by)
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        batches = self.handle_singleton(batches)

        if self.shuffle:
            np.random.shuffle(batches)

        batch_idx = 0
        while batch_idx < len(batches) - 1:
            yield batches[batch_idx]
            batch_idx += 1
        if len(batches) > 0 and (len(batches[batch_idx]) == self.batch_size or not self.drop_last):
            yield batches[batch_idx]

    def __len__(self):
        if self.drop_last:
            return math.floor(self.data_size / self.batch_size)
        else:
            return math.ceil(self.data_size / self.batch_size)
