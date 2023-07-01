# IDRR-Num

This repo contains the PyTorch implementation of Numerical Semantic Modeling For Implicit Discourse Relation Recognition ICASSP 2023.

## Data

We use PDTB 2.0 and CoNLL 2016 Shared Task to evaluate our models. If you have bought data from LDC, please put the PDTB data and CoNLL data in *data/pdtb* and *data/conll* respectively.

## Preprocessing

### Numerical Preprocessing

- In the directory of *src*, run the following command to get txt file:

```
python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_file_path ../data/pdtb/train_pdtb.txt \
    --sections 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_file_path ../data/pdtb/valid_pdtb.txt \
    --sections 0,1

python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_path ../data/pdtb/test_pdtb.txt \
    --sections 21,22
```

```
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-train/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_file_path ../data/conll/train_conll.txt
    
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-dev/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_file_path ../data/conll/valid_conll.txt
    
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-test/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_file_path ../data/conll/test_conll.txt
    
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-blind-test/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type txt \
    --dataset_file_path ../data/conll/blind_conll.txt
```

- Download the CoreNLP tools from https://stanfordnlp.github.io/CoreNLP/.

- In the directory of the CoreNLP tools, run the following command:

```
java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -ssplit.eolonly -file input.txt -outputFormat json
```

input.txt should be train_pdtb/dev_pdtb/test_pdtb/train_conll/dev_conll/test_conll/blind_conll.txt, then you will get the output with a JSON extension.

- In the directory of *src*, run the following command to get json file:

```
python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/pdtb/train_pdtb.json \
    --sections 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/pdtb/valid_pdtb.json \
    --sections 0,1

python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/pdtb/test_pdtb.json \
    --sections 21,22
```

```
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-train/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/conll/train_conll.json
    
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-dev/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/conll/valid_conll.json
    
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-test/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/conll/test_conll.json
    
python preprocess_conll.py \
    --json_file_path ../data/conll/conll16st-en-03-29-16-blind-test/relations.json \
    --types Implicit \
    --encoder roberta \
    --preprocess_type json \
    --dataset_file_path ../data/conll/blind_conll.json
```

- In the directory of *src*, run the following command to get final numerical datasets file:

```
python preprocess_pdtb.py \
    --origin_file_path ../data/pdtb/train_pdtb.json \
    --json_ner_file_path ../data/pdtb/train_pdtb.txt.json \
    --dataset_file_path ../data/pdtb/train_pdtb_numerical.pt 
    
python preprocess_pdtb.py \
    --origin_file_path ../data/pdtb/valid_pdtb.json \
    --json_ner_file_path ../data/pdtb/valid_pdtb.txt.json \
    --dataset_file_path ../data/pdtb/valid_pdtb_numerical.pt 
    
python preprocess_pdtb.py \
    --origin_file_path ../data/pdtb/test_pdtb.json \
    --json_ner_file_path ../data/pdtb/test_pdtb.txt.json \
    --dataset_file_path ../data/pdtb/test_pdtb_numerical.pt 
```

```
python preprocess_conll.py \
    --origin_file_path ../data/pdtb/train_conll.json \
    --json_ner_file_path ../data/pdtb/train_conll.txt.json \
    --dataset_file_path ../data/pdtb/train_conll_numerical.pt 
    
python preprocess_conll.py \
    --origin_file_path ../data/pdtb/valid_conll.json \
    --json_ner_file_path ../data/pdtb/valid_conll.txt.json \
    --dataset_file_path ../data/pdtb/valid_conll_numerical.pt 
    
python preprocess_conll.py \
    --origin_file_path ../data/pdtb/test_conll.json \
    --json_ner_file_path ../data/pdtb/test_conll.txt.json \
    --dataset_file_path ../data/pdtb/test_conll_numerical.pt
        
python preprocess_conll.py \
    --origin_file_path ../data/pdtb/blind_conll.json \
    --json_ner_file_path ../data/pdtb/blind_conll.txt.json \
    --dataset_file_path ../data/pdtb/blind_conll_numerical.pt
```

## Training

For PDTB, run the following command:

```
sh run_pdtb.sh
```

For CoNLL, run the following command:

```
sh run_conll.sh
```

## Bibliography

If you find this repo useful, please cite our paper.

```
@inproceedings{wang2023numerical,
  title={Numerical Semantic Modeling for Implicit Discourse Relation Recognition},
  author={Wang, Chenxu and Jian, Ping and Wang, Hai},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```