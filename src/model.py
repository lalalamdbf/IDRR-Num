import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from encoder import *
from tools import allennlp as util
from util import masked_mean, GCN



class NumericalModel(nn.Module):
    def __init__(self, **kw):
        super(NumericalModel, self).__init__()
        encoder = kw.get("encoder", "roberta")
        dropout = kw.get("dropout", 0.2)
        self.use_gcn = kw.get("use_gcn", True)
        gcn_steps = kw.get("gcn_steps", 1)
        num_rels = kw.get("num_rels", 4)
        self.drop = nn.Dropout(dropout)
        node_dim = 768

        if encoder == "roberta":
            self.encoder = ROBERTAEncoder(**kw)
        elif encoder == "bert":
            self.encoder = BERTEncoder(**kw)
        else:
            raise NotImplementedError(
                "Error: encoder=%s is not supported now." % (encoder))

        if self.use_gcn:
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._iteration_steps = gcn_steps

        self.classifier = nn.Linear(node_dim, num_rels)



    def forward(self, arg1, arg2, arg1_mask=None, arg2_mask=None, arg1_number_indices = None, arg1_number_order = None, arg1_ner_type = None, arg2_number_indices= None, arg2_number_order= None, arg2_ner_type= None):
        
        outputs, arg1_mask, arg2_mask = self.encoder.forward(
                arg1, arg2, arg1_mask, arg2_mask)
        
        sequence_output = outputs
        
        if self.use_gcn:
            batch_size = arg1.size(0)

            sequence_alg = outputs
            encoded_arg1_for_numbers = sequence_alg
            encoded_arg2_for_numbers = sequence_alg

            # arg1 number extraction
            arg1_number_mask = (arg1_number_indices > -1).long()
            clamped_arg1_number_indices = util.replace_masked_values(
                arg1_number_indices, arg1_number_mask, 0)
            arg1_encoded_number = torch.gather(encoded_arg1_for_numbers, 1,
                                                clamped_arg1_number_indices.unsqueeze(-1).expand(-1, -1, encoded_arg1_for_numbers.size(-1)))

            # arg2 number extraction
            arg2_number_mask = (arg2_number_indices > -1).long()
            clamped_arg2_number_indices = util.replace_masked_values(
                arg2_number_indices, arg2_number_mask, 0)
            arg2_encoded_number = torch.gather(encoded_arg2_for_numbers, 1,
                                            clamped_arg2_number_indices.unsqueeze(-1).expand(-1, -1, encoded_arg2_for_numbers.size(-1)))
            # graph mask
            all_ner_type = torch.cat([arg1_ner_type,arg2_ner_type],-1)
            ner_mask = (all_ner_type.unsqueeze(-1).expand(-1,-1,
                                                        all_ner_type.size(-1)) == all_ner_type.unsqueeze(1).expand(-1, all_ner_type.size(-1), -1)).long() 
            number_order = torch.cat(
                (arg1_number_order, arg2_number_order), -1)
            new_graph_mask = number_order.unsqueeze(1).expand(batch_size, number_order.size(-1),
                                                            -1) > number_order.unsqueeze(-1).expand(batch_size, -1, number_order.size(-1))
            new_graph_mask = new_graph_mask.long()
            all_number_mask = torch.cat(
                (arg1_number_mask, arg2_number_mask), dim=-1)
            new_graph_mask = all_number_mask.unsqueeze(
                1) * all_number_mask.unsqueeze(-1) * new_graph_mask
            d_node, q_node = self._gcn(d_node=arg1_encoded_number, q_node=arg2_encoded_number,
                                                        d_node_mask=arg1_number_mask, q_node_mask=arg2_number_mask, ner_mask=ner_mask ,graph=new_graph_mask)
            
            all_number_node = torch.cat((d_node,q_node),dim=-2)
            
            gcn_info_vec = torch.zeros((batch_size, sequence_alg.size(1) + 1, sequence_output.size(-1)),
                                    dtype=torch.float, device=all_number_node.device)
            
            all_number_indices = torch.cat((arg1_number_indices, arg2_number_indices),dim=-1)
            
            clamped_number_indices = util.replace_masked_values(all_number_indices, all_number_mask,
                                                                    gcn_info_vec.size(1) - 1)
            gcn_info_vec.scatter_(
                1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, all_number_node.size(-1)), all_number_node)
            gcn_info_vec = gcn_info_vec[:, :-1, :]
            
            sequence_output = sequence_output + gcn_info_vec
            
        mask = torch.cat(
            (arg1_mask, arg2_mask), dim=-1)
        mul_mask = lambda x, m: x * torch.unsqueeze(m, dim=-1)
        masked_reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (
                torch.sum(m, dim=1, keepdims=True) + 1e-10)

        
        mean_output = masked_reduce_mean(sequence_output, mask)
        mean_output = self.drop(mean_output)
        
        output = self.classifier(mean_output)

        return output  
