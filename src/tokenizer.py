import os
import torch
import torch.nn as nn
import re
from itertools import chain
from transformers import BertTokenizer, RobertaTokenizer
from util import *

class Tokenizer:
    def __init__(self, **kw):
        pass
    
    def encode(self, sentence, return_tensors=False):
        raise NotImplementedError

    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        raise NotImplementedError

    @property
    def pad_token_id(self):
        raise NotImplementedError

    def concat_sent_ids(self, sent_ids):
        raise NotImplementedError

class BERTTokenizer(Tokenizer):
    def __init__(self, **kw):
        super(BERTTokenizer, self).__init__(**kw)
        corenlp_path = kw.get("corenlp_path", "")
        corenlp_port = kw.get("corenlp_port", 0)
        bert_dir = kw.get("bert_dir", "../data/pretrained_lm/bert")

        self.corenlp_client = get_corenlp_client(corenlp_path=corenlp_path, corenlp_port=corenlp_port)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=bert_dir)
    
    def encode(self, sentence, return_tensors=False):
        """
        :param sentence: a string of sentence
        :return encoded_result: a list of ids or a tensor if return_tensors=True
        """
        ids = self.tokenizer.encode(sentence)

        if return_tensors:
            ids = torch.tensor(ids)
        return ids
    
    def get_tokenizer(self):
        return self.tokenizer
          
    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        """
        :param batch_sentences: a string of sentences or a list of sentences
        :return encoded_result: a list of lists of ids or a tensor if return_tensors=True and masks
        """
        if isinstance(batch_sentences, str):
            batch_sentences = sentence_split_with_corenlp(batch_sentences, self.corenlp_client)

        batch_outputs = self.tokenizer.batch_encode_plus(batch_sentences, 
            return_tensors="pt" if return_tensors else None,
            return_input_lengths=return_lengths,
            return_attention_masks=return_masks)
        results = dict({"ids": batch_outputs["input_ids"]})
        if return_lengths:
            results["lens"] = batch_outputs["input_len"]
        if return_masks:
            results["masks"] = batch_outputs["attention_mask"]
        return results

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def concat_sent_ids(self, sent_ids):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        ids = list()
        if isinstance(sent_ids[0], list):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.extend(sent_id)
                else:
                    ids.extend(sent_id[1:])
        elif isinstance(sent_ids[0], torch.Tensor):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.append(sent_id)
                else:
                    ids.append(sent_id[1:])
            ids = torch.cat(ids, dim=-1)
        else:
            raise ValueError
        return ids
    
class ROBERTATokenizer(Tokenizer):
    def __init__(self, **kw):
        super(ROBERTATokenizer, self).__init__(**kw)

        self.tokenizer = RobertaTokenizer.from_pretrained("../data/pretrained_lm/roberta")
    
    def encode(self, sentence, return_tensors=False):
        """
        :param sentence: a string of sentence
        :return encoded_result: a list of ids or a tensor if return_tensors=True
        """
        ids = self.tokenizer.encode(sentence)

        if return_tensors:
            ids = torch.tensor(ids)
        return ids
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def batch_encode(self, batch_sentences, return_tensors=False, return_lengths=False, return_masks=False):
        """
        :param batch_sentences: a string of sentences or a list of sentences
        :return encoded_result: a list of lists of ids or a tensor if return_tensors=True and masks
        """
        if isinstance(batch_sentences, str):
            batch_sentences = sentence_split_with_corenlp(batch_sentences, self.corenlp_client)

        batch_outputs = self.tokenizer.batch_encode_plus(batch_sentences, 
            return_tensors="pt" if return_tensors else None,
            return_input_lengths=return_lengths,
            return_attention_masks=return_masks)
        results = dict({"ids": batch_outputs["input_ids"]})
        if return_lengths:
            results["lens"] = batch_outputs["input_len"]
        if return_masks:
            results["masks"] = batch_outputs["attention_mask"]
        return results

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def concat_sent_ids(self, sent_ids):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
        ids = list()
        sep = self.tokenizer.sep_token_id
        if isinstance(sent_ids[0], list):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.extend(sent_id)
                else:
                    ids.append(sep)
                    ids.extend(sent_id[1:])
        elif isinstance(sent_ids[0], torch.Tensor):
            for i, sent_id in enumerate(sent_ids):
                if i == 0:
                    ids.append(sent_id)
                else:
                    sent_id_ = sent_id.clone()
                    sent_id_[0].fill_(sep)
                    ids.append(sent_id_)
            ids = torch.cat(ids, dim=-1)
        else:
            raise ValueError
        return ids
