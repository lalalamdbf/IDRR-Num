import os
import socket
import re
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from stanfordnlp.server import CoreNLPClient
from tools import allennlp as util

def masked_mean(vector, mask):
    mask1 = mask.unsqueeze(-1)
    replaced_vector = vector.masked_fill(mask1==0, 0.0) if mask1 is not None else vector
    value_sum = replaced_vector.sum(-2)
    value_count = mask.sum(-1).unsqueeze(-1)
    return value_sum / value_count
##############################
##### argument functions #####
##############################
def str2value(x):
    return eval(x)

def str2bool(x):
    x = x.lower()
    return x == "true" or x == "yes"

def str2list(x):
    results = []
    for x in x.split(","):
        x = x.strip()
        try:
            x = str2value(x)
        except:
            pass
        results.append(x)
    return results

##############################
###### stanford corenlp ######
##############################
def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path, corenlp_port):
    annotators = ["tokenize", "ssplit"]

    os.environ["CORENLP_HOME"] = corenlp_path
    if is_port_occupied(port=corenlp_port):
        try:
            corenlp_client = CoreNLPClient(
                annotators=annotators, timeout=99999,
                memory='4G', endpoint="http://localhost:%d" % corenlp_port,
                start_server=False, be_quiet=False)
            return corenlp_client
        except Exception as err:
            raise err
    else:
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators, timeout=99999,
            memory='4G', endpoint="http://localhost:%d" % corenlp_port,
            start_server=True, be_quiet=False)
        return corenlp_client

def sentence_split_with_corenlp(sentence, corenlp_client):
    results = list()
    while len(results) == 0:
        try:
            for sent in corenlp_client.annotate(sentence, annotators=["ssplit"], output_format="json")["sentences"]:
                if sent['tokens']:
                    char_st = sent['tokens'][0]['characterOffsetBegin']
                    char_end = sent['tokens'][-1]['characterOffsetEnd']
                else:
                    char_st, char_end = 0, 0
                results.append(sentence[char_st:char_end])
            break
        except:
            pass
    return results

def tokenize_with_corenlp(sentence, corenlp_client):
    results = list()
    
    while len(results) == 0:
        try:
            for sent in corenlp_client.annotate(sentence, annotators=["ssplit"], output_format="json")["sentences"]:
                results.append([t['word'] for t in sent['tokens']])
            break
        except:
            pass
    return results


##############################
######## os functions ########
##############################
def save_config(config, path):
    with open(path, "w") as f:
        json.dump(vars(config), f)

def load_config(path):
    with open(path, "r") as f:
        config = json.load(f, object_hook=lambda d: namedtuple('config', d.keys())(*d.values()))
    return config

def _map_tensor_to_list(tensor):
    return tensor.tolist()

def _map_array_to_list(array):
    return array.tolist()

def _map_list_to_python_type(l):
    if len(l) == 0:
        return l
    if isinstance(l[0], dict):
        return [_map_dict_to_python_type(x) for x in l]
    elif isinstance(l[0], list):
        return [_map_list_to_python_type(x) for x in l]
    elif isinstance(l[0], torch.Tensor):
        return [_map_tensor_to_list(x) for x in l]
    elif isinstance(l[0], np.ndarray):
        return [_map_array_to_list(x) for x in l]
    else:
        return l

def _map_dict_to_python_type(d):
    new_d = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = _map_dict_to_python_type(v)
        elif isinstance(v, list):
            new_d[k] = _map_list_to_python_type(v)
        elif isinstance(v, torch.Tensor):
            new_d[k] = _map_tensor_to_list(v)
        elif isinstance(v, np.ndarray):
            new_d[k] = _map_array_to_list(v)
        else:
            new_d[k] = v
    return new_d

def save_results(results, path):
    with open(path, "w") as f:
        json.dump(_map_dict_to_python_type(results), f)

def get_best_epochs(log_file, by="loss"):
    regex = re.compile(r"data_type:\s+(\w+)\s+best\s+([\w\-]+).*?\(epoch:\s+(\d+)\)")
    best_epochs = dict()
    # get the best epoch
    try:
        lines = subprocess.check_output(["tail", log_file, "-n12"]).decode("utf-8").split("\n")[0:-1]
    except:
        with open(log_file, "r") as f:
            lines = f.readlines()
    
    for line in lines[-12:]:
        matched_results = regex.findall(line)
        for matched_result in matched_results:
            if by in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    if len(best_epochs) == 0:
        for line in lines:
            matched_results = regex.findall(line)
            if by in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    return best_epochs

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


##############################
### deep learning functions ##
##############################
def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = len(graph_sizes)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = min(graph_sizes), max(graph_sizes)
    mask = torch.ones((bsz, max_size), dtype=torch.long, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes, dim=0))
        for i, l in enumerate(graph_sizes):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask

def batch_convert_list_to_tensor(batch_list, max_seq_len=-1):
    batch_tensor = [torch.tensor(v) for v in batch_list]
    return batch_convert_tensor_to_tensor(batch_tensor)

def batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=-1):
    batch_lens = [len(v) for v in batch_tensor]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)

    result = torch.ones([len(batch_tensor), max_seq_len] + list(batch_tensor[0].size())[1:], dtype=batch_tensor[0].dtype, requires_grad=False)
    for i, t in enumerate(batch_tensor):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            result[i, :len_t].data.copy_(t)
        elif len_t == max_seq_len:
            result[i].data.copy_(t)
        else:
            result[i].data.copy_(t[:max_seq_len])
    return result

def batch_convert_len_to_mask(batch_lens, max_seq_len=-1):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = torch.ones((len(batch_lens), max_seq_len), dtype=torch.float, requires_grad=False)
    for i, l in enumerate(batch_lens):
        mask[i, l:].fill_(0)
    return mask

class _GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

_act_map = {"none": None,
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(1/5.5),
            "prelu": nn.PReLU()}
_act_map["gelu"] = _GELU()

def map_activation_str_to_layer(act_str):
    try:
        return _act_map[act_str]
    except:
        raise NotImplementedError("Error: %s activation fuction is not supported now." % (act_str))

def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1-lambda0)/(1+np.exp(-K*(t-T/2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1-lambda0) * t/T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1-lambda0) * (1 - math.cos(math.pi * t/T))/2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t-R*T, R*T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError

def change_dropout_rate(model, dropout):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            child.p = dropout
        change_dropout_rate(child, dropout)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)



class GCN(nn.Module):

    def __init__(self, node_dim, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps
         
        self._d_node_query = torch.nn.Linear(node_dim, node_dim, bias=False) # d_node query
        self._d_node_key = torch.nn.Linear(node_dim, node_dim, bias=False) # d_node key
        self._q_node_query = torch.nn.Linear(node_dim, node_dim, bias=False) # q_node query
        self._q_node_key = torch.nn.Linear(node_dim, node_dim, bias=False) # q_node key
        
        self._dq_node_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc = torch.nn.Linear(node_dim, node_dim, bias=False)
        

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        

    def forward(self, d_node, q_node, d_node_mask, q_node_mask, ner_mask, graph):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        d_mask = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * ner_mask[:, :d_node_len, :d_node_len]
        # d_mask = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) 
        dd_graph = d_mask * (1 - diagmat) 
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        q_mask = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * ner_mask[:, d_node_len:, d_node_len:]
        qq_graph = q_mask  * (1 - diagmat)  
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1) * ner_mask[:, :d_node_len, d_node_len:]
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1) * ner_mask[:, d_node_len:, :d_node_len] 
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])


        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(-1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(-1) + qd_graph_right.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)
        
        d_k = d_node.size(-1)


        for step in range(self.iteration_steps):
            
            d_node_query = self._d_node_query(d_node)
            d_node_key = self._d_node_key(d_node)
            d_scores = torch.matmul(d_node_query,d_node_key.transpose(-2,-1)) / math.sqrt(d_k)
            d_scores = d_scores.masked_fill(d_mask == 0, -1e9)
            d_attn = d_scores.softmax(dim=-1)

            q_node_query = self._q_node_query(q_node)
            q_node_key = self._q_node_key(q_node)
            q_scores = torch.matmul(q_node_query,q_node_key.transpose(-2,-1)) / math.sqrt(d_k)
            q_scores = q_scores.masked_fill(q_mask == 0, -1e9)
            q_attn = q_scores.softmax(dim=-1)
            
            dq_scores = torch.matmul(d_node,self._dq_node_fc(q_node).transpose(-2,-1))
            dq_scores = dq_scores.masked_fill(dq_graph == 0, -1e9)
            dq_attn = dq_scores.softmax(dim=-1)
            
            qd_scores = torch.matmul(q_node,self._qd_node_fc(d_node).transpose(-2,-1))
            qd_scores = qd_scores.masked_fill(qd_graph == 0, -1e9)
            qd_attn = qd_scores.softmax(dim=-1)

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_attn,
                    dd_graph_left,
                    0)
            
            qd_node_weight = util.replace_masked_values(
                    qd_attn,
                    qd_graph_left,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_attn,
                    qq_graph_left,
                    0)

            dq_node_weight = util.replace_masked_values(
                    dq_attn,
                    dq_graph_left,
                    0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)
            
            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

           
            dd_node_weight = util.replace_masked_values(
                    d_attn,
                    dd_graph_right,
                    0)

            qd_node_weight = util.replace_masked_values(
                    qd_attn,
                    qd_graph_right,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_attn,
                    qq_graph_right,
                    0)

            dq_node_weight = util.replace_masked_values(
                    dq_attn,
                    dq_graph_right,
                    0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)
           


            agg_d_node_info = (dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(-1)
            
            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)

        return d_node, q_node 
    

