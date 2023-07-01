import argparse
import os
import pandas as pd

import json
from functools import partial
from tokenizer import *
from dataset import Dataset
from util import str2list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Raw Data
    parser.add_argument("--csv_file_path", type=str, default="",
                        help="pdtb2.csv location")
  
    # Sections to be processed
    parser.add_argument("--sections", type=str2list, default=list(),
                        help="sections")
    # Types to be processed
    parser.add_argument("--types", type=str2list, default=["Implicit"],
                        help="relation types")

    # Encoder
    parser.add_argument("--encoder", type=str, default="roberta",
                        choices=["bert", "roberta"],
                        help="the encoder")
    
    # Encoder
    parser.add_argument("--preprocess_type", type=str, default="json",
                        choices=["json", "txt", "numerical"],
                        help="the encoder")

    
    # json dataset used in get_numerical_dataset
    parser.add_argument("--original_file_path", type=str, default="")
      
    # json ner dataset used in get_numerical_dataset
    parser.add_argument("--json_ner_file_path", type=str, default="") 
    
    # Saved dataset
    parser.add_argument("--dataset_file_path", type=str, default="",
                        help="dataset location")     
    
    # Log
    parser.add_argument("--log_path", type=str, default="./preprocess_pdtb.log",
                        help="log path of pdtb output")

    args = parser.parse_args()

    args.encoder = args.encoder.lower()
   
    if args.encoder == "roberta":
        tokenizer = ROBERTATokenizer(**vars(args))
    elif args.encoder == "bert":
        tokenizer = BERTTokenizer(**vars(args))
    else:
        raise NotImplementedError("Error: encoder=%s is not supported now." % (args.encoder))

    
    # Dataset
    pdtb_dataset = Dataset(data=None, tokenizer=tokenizer.get_tokenizer())
    
    if args.preprocess_type == 'json':
        # get original dataset json
        pdtb_dataset.get_dataset_pdtb(args.csv_file_path, args.sections, args.types, is_tokenizer= False)
        pdtb_dataset.save_json(args.dataset_file_path)
    elif args.preprocess_type == 'txt':
        # get original dataset txt
        pdtb_dataset.get_txt_pdtb(args.csv_file_path, args.dataset_file_path, args.sections, args.types, is_tokenizer= False)
    elif args.preprocess_type == 'numerical':
        # get numerical dataset
        pdtb_dataset.get_numerical_dataset(args.original_file_path, args.json_ner_file_path)
        pdtb_dataset.save_pt(args.dataset_file_path)
    else:
        raise NotImplementedError("Error: preprocess_type=%s is not supported now." % (args.preprocess_type))
    
        

