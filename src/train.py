import os
from xmlrpc.client import boolean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import copy
import argparse
import logging
import gc
import datetime
import pprint
from collections import OrderedDict
from functools import partial
from torch.optim import Adam
from torch.optim import AdamW
from transformers import WarmUp
try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from model import NumericalModel
from dataset import Dataset, Sampler
from evaluate import evaluate_accuracy, evaluate_precision_recall_f1
from util import *

INF = 1e30
_INF = -1e30

def eval_epoch(args, logger, writer, model, data_type, data_loader, device, epoch):
    model.eval()
    epoch_step = len(data_loader)
    total_step = args.epochs * epoch_step
    total_cnt = 0
    total_loss = 0.0

    results = {"data": {"id": list(), "relation": list(), "prefered_relation": list()},
        "prediction": {"prob": list(), "pred": list()},
        "evaluation": {"accuracy": dict(), "precision_recall_f1": dict()}}

    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            step = epoch*epoch_step+batch_id
            _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation, arg1_number_indices, arg1_number_order, \
            arg1_ner_type, arg2_number_indices, arg2_number_order, arg2_ner_type= batch
            bsz = len(_id)
            total_cnt += bsz

            results["data"]["id"].extend(_id)
            results["data"]["relation"].extend(relation)
            results["data"]["prefered_relation"].extend(prefered_relation)

            arg1 = arg1.to(device)
            arg2 = arg2.to(device)
            if arg1_mask is not None:
                arg1_mask = arg1_mask.to(device)
            if arg2_mask is not None:
                arg2_mask = arg2_mask.to(device)
            relation = relation.to(device)
            prefered_relation = prefered_relation.to(device)
            arg1_number_indices = arg1_number_indices.to(device)
            arg1_number_order = arg1_number_order.to(device)
            arg1_ner_type = arg1_ner_type.to(device)
            arg2_number_indices = arg2_number_indices.to(device)
            arg2_number_order = arg2_number_order.to(device)
            arg2_ner_type = arg2_ner_type.to(device)
            
            output = model(arg1, arg2, arg1_mask, arg2_mask, arg1_number_indices, arg1_number_order,  arg1_ner_type, arg2_number_indices, arg2_number_order,  arg2_ner_type)
            # logp = F.log_softmax(output, dim=1)
            # prob = logp.exp()
            prob = torch.nn.functional.softmax(output, dim=-1)
            results["prediction"]["prob"].extend(prob.cpu().detach())
            results["prediction"]["pred"].extend(prob.cpu().argmax(dim=1).detach())
            
            loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_cross_entropy(output, relation)
            # loss = loss_cross_entropy(logp, relation)
            avg_loss = loss
            total_loss += avg_loss.item() * bsz

            if writer:
                writer.add_scalar("%s/pdtb-loss" % (data_type), avg_loss.item(), step)

            if logger and batch_id == epoch_step-1:
                logger.info(
                    "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}".format(
                        epoch, args.epochs, data_type, batch_id, epoch_step) + "\n" +
                    "\tpdtb-loss: {:10.4f}".format(
                        avg_loss.item()) + "\n" +
                    "\tpdtb-gold: {}".format(results["data"]["relation"][-1]) + "\n" + 
                    "\tpdtb-pred: {}".format(results["prediction"]["prob"][-1]))

        mean_loss = total_loss / total_cnt 

        pred = np.array(results["prediction"]["pred"])
        target = torch.cat(results["data"]["relation"], dim=0).view(total_cnt, -1).int().numpy()
        prefered_target = np.array(results["data"]["prefered_relation"])

        
        results["evaluation"]["accuracy"] = evaluate_accuracy(pred, target, prefered_target)
        results["evaluation"]["precision_recall_f1"] = evaluate_precision_recall_f1(pred, target, prefered_target)

        if writer:
            writer.add_scalar("%s/pdtb-loss-epoch" % (data_type), mean_loss, epoch)

        if logger:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}".format(
                    epoch, args.epochs, data_type) + "\n" +
                "\tpdtb-loss-epoch: {:10.4f}".format(
                    mean_loss) + "\n" +
                "\tpdtb-accuray: {}".format(
                    pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n\t\t")) + "\n" +
                "\tpdtb-precision_recall_f1: {}".format(
                    pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n\t\t")))
    gc.collect()
    return mean_loss, results

def train_epoch(args, logger, writer, model, optimizer, data_type, data_loader, device, epoch):
    model.train()
    epoch_step = len(data_loader)
    total_step = args.epochs * epoch_step
    total_cnt = 0
    total_loss = 0.0

    results = {"data": {"id": list(), "relation": list(), "prefered_relation": list()},
        "prediction": {"prob": list(), "pred": list()},
        "evaluation": {"accuracy": dict(), "precision_recall_f1": dict()}}

    for batch_id, batch in enumerate(data_loader):
        step = epoch*epoch_step+batch_id
        _id, arg1, arg1_mask, arg2, arg2_mask, relation, prefered_relation, arg1_number_indices, arg1_number_order, \
        arg1_ner_type, arg2_number_indices, arg2_number_order, arg2_ner_type= batch
        bsz = len(_id)
        total_cnt += bsz

        results["data"]["id"].extend(_id)
        results["data"]["relation"].extend(relation)
        results["data"]["prefered_relation"].extend(prefered_relation)

        arg1 = arg1.to(device)
        arg2 = arg2.to(device)
        if arg1_mask is not None:
            arg1_mask = arg1_mask.to(device)
        if arg2_mask is not None:
            arg2_mask = arg2_mask.to(device)
        relation = relation.to(device)
        prefered_relation = prefered_relation.to(device)
        arg1_number_indices = arg1_number_indices.to(device)
        arg1_number_order = arg1_number_order.to(device)
        arg1_ner_type = arg1_ner_type.to(device)
        arg2_number_indices = arg2_number_indices.to(device)
        arg2_number_order = arg2_number_order.to(device)
        arg2_ner_type = arg2_ner_type.to(device)
        
        output = model(arg1, arg2, arg1_mask, arg2_mask, arg1_number_indices, arg1_number_order,  arg1_ner_type, arg2_number_indices, arg2_number_order,  arg2_ner_type)
        # logp = F.log_softmax(output, dim=1)
        # prob = logp.exp()
        prob = torch.nn.functional.softmax(output, dim=-1)
        results["prediction"]["prob"].extend(prob.cpu().detach())
        results["prediction"]["pred"].extend(prob.cpu().argmax(dim=1).detach())
        
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_cross_entropy(output, relation)
        # loss = loss_cross_entropy(logp, relation)
        avg_loss = loss

        total_loss += avg_loss.item() * bsz
        
        avg_loss.backward()
      
        optimizer.step()
        model.zero_grad()

        if writer:
            writer.add_scalar("%s/pdtb-loss" % (data_type), avg_loss.item(), step)

        if logger and (batch_id%args.print_every == 0 or batch_id == epoch_step-1):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}".format(
                    epoch, args.epochs, data_type, batch_id, epoch_step) + "\n" +
                "\tpdtb-loss: {:10.4f}".format(
                    avg_loss.item()) + "\n" +
                "\tpdtb-gold: {}".format(results["data"]["relation"][-1]) + "\n" + 
                "\tpdtb-pred: {}".format(results["prediction"]["prob"][-1]))

    mean_loss = total_loss / total_cnt

    pred = np.array(results["prediction"]["pred"])
    target = torch.cat(results["data"]["relation"], dim=0).view(total_cnt, -1).int().numpy()
    prefered_relation = np.array(results["data"]["prefered_relation"])


    results["evaluation"]["accuracy"] = evaluate_accuracy(pred, target, prefered_relation)
    results["evaluation"]["precision_recall_f1"] = evaluate_precision_recall_f1(pred, target, prefered_relation)

    if writer:
        writer.add_scalar("%s/pdtb-loss-epoch" % (data_type), mean_loss, epoch)

    if logger:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}".format(
                epoch, args.epochs, data_type) + "\n" +
            "\tpdtb-loss-epoch: {:10.4f}".format(
                mean_loss) + "\n" +
            "\tpdtb-accuray: {}".format(
                pprint.pformat(results["evaluation"]["accuracy"]).replace("\n", "\n\t\t")) + "\n" +
            "\tpdtb-precision_recall_f1: {}".format(
                pprint.pformat(results["evaluation"]["precision_recall_f1"]).replace("\n", "\n\t\t")))
                
    gc.collect()
    return mean_loss, results

def train(args, logger, writer):
    # set device
    if args.gpu_ids is None:
        device = torch.device("cpu")
    else:
        if isinstance(args.gpu_ids, int):
            args.gpu_ids = [args.gpu_ids]
        device = torch.device("cuda:%d" % args.gpu_ids[0])
        torch.cuda.set_device(device)

    if args.pretrained_model_path:
        # load pretrained model
        config = load_config(os.path.join(args.pretrained_model_path, "NumericalModel.config"))
        for by in ["accf1", "f1", "accuracy", "loss"]:
            best_epochs = get_best_epochs(os.path.join(args.pretrained_model_path, "NumericalModel.log"), by=by)
            if len(best_epochs) > 0:
                break
        logger.info("retrieve the best epochs for NumericalModel: %s" % (best_epochs))
        if len(best_epochs) > 0:
            model = NumericalModel(**(config._asdict()))
            if "valid" in best_epochs:
                model.load_state_dict(torch.load(
                    os.path.join(args.pretrained_model_path, "epoch%d.pt" % (best_epochs["test"])),
                    map_location=device))
            if config.dropout != args.dropout:
                change_dropout_rate(model, args.dropout)
        else:
            raise ValueError("Error: cannot load NumericalModel from %s." % (args.pretrained_model_path))
    else:
        # build model
        model = NumericalModel(**vars(args))

    if args.gpu_ids and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model = model.to(device)
    logger.info(model)
    logger.info("num of trainable parameters: %d" % (
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    datasets = OrderedDict({
        "train": Dataset().load_pt(args.train_dataset_path), 
        "valid": Dataset().load_pt(args.valid_dataset_path), 
        "test": Dataset().load_pt(args.test_dataset_path)})
    if args.blind_dataset_path != "":
        blind_dataset = Dataset().load_pt(args.blind_dataset_path)
        datasets["blind"] = blind_dataset
    for data_type, dataset in datasets.items():
        for x in dataset:
            x["arg1_len"] = len(x["arg1"])
            x["arg2_len"] = len(x["arg2"])

    if args.blind_dataset_path == "":
        logger.info("train:valid:test = %d:%d:%d" % (len(datasets["train"]), len(datasets["valid"]), len(datasets["test"])))
    else:
        logger.info("train:valid:test = %d:%d:%d:%d" % (len(datasets["train"]), len(datasets["valid"]), len(datasets["test"]), len(datasets["blind"])))
    if args.num_rels == 14:
        rel_map = Dataset.rel_map_14
    elif args.num_rels == 11:
        rel_map = Dataset.rel_map_11
    elif args.num_rels == 4:
        rel_map = Dataset.rel_map_4
    else:
        raise NotImplementedError("Error: num_rels=%d is not supported now." % (args.num_rels))
    if args.encoder == "roberta":
        pad_id = 1
    else:
        pad_id = 0
    data_loaders = OrderedDict()
    batchify = partial(Dataset.batchify_numerical,
        rel_map=rel_map, min_arg=args.min_arg, max_arg=args.max_arg, pad_id=pad_id)
    for data_type in datasets:
        sampler = Sampler(datasets[data_type], 
            group_by=["arg1_len", "arg2_len"], batch_size=args.batch_size, 
            shuffle=data_type=="train", drop_last=False)
        data_loaders[data_type] = data.DataLoader(datasets[data_type], 
            batch_sampler=sampler, 
            collate_fn=batchify, 
            pin_memory=data_type=="train")
    
    
    
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]},
    {'params': [p for n, p in model.named_parameters() if 'encoder' not in n],'lr': args.downstream_lr}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer.zero_grad()
    best_losses = {dataset: INF for dataset in datasets}
    best_loss_epochs = {dataset: -1 for dataset in datasets}
    best_accs = {dataset: _INF for dataset in datasets}
    best_acc_epochs = {dataset: -1 for dataset in datasets}
    best_f1s = {dataset: _INF for dataset in datasets}
    best_f1_epochs = {dataset: -1 for dataset in datasets}
    best_accf1s = {dataset: _INF for dataset in datasets}
    best_accf1_epochs = {dataset: -1 for dataset in datasets}

    for epoch in range(args.epochs):
        for data_type, data_loader in data_loaders.items():
            
            if data_type == "test" or data_type == "blind":
                continue
            
            if data_type == "train":
                mean_loss, results = train_epoch(args, logger, writer,
                    model, optimizer, data_type, data_loader, device, epoch)
            else:
                mean_loss, results = eval_epoch(args, logger, writer,
                    model, data_type, data_loader, device, epoch)
                # save_results(results, os.path.join(args.save_model_dir, "%s_results%d.json" % (data_type, epoch)))

            if mean_loss <= best_losses[data_type]:
                best_losses[data_type] = mean_loss
                best_loss_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-loss: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_losses[data_type], best_loss_epochs[data_type]))
                if args.save_best == "loss":
                    if data_type == "valid":
                        if args.gpu_ids and len(args.gpu_ids) > 1:
                            torch.save(model.module.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
                        else:
                            torch.save(model.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
            if results["evaluation"]["accuracy"]["overall"] >= best_accs[data_type]:
                best_accs[data_type] = results["evaluation"]["accuracy"]["overall"] 
                best_acc_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-accuracy: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_accs[data_type], best_acc_epochs[data_type]))
                if args.save_best == "acc":
                    if data_type == "valid":
                        if args.gpu_ids and len(args.gpu_ids) > 1:
                            torch.save(model.module.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
                        else:
                            torch.save(model.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
            if results["evaluation"]["precision_recall_f1"]["overall"][-1] >= best_f1s[data_type]:
                best_f1s[data_type] = results["evaluation"]["precision_recall_f1"]["overall"][-1]
                best_f1_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-f1: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_f1s[data_type], best_f1_epochs[data_type]))
                if args.save_best == "f1":
                    if data_type == "valid":
                        if args.gpu_ids and len(args.gpu_ids) > 1:
                            torch.save(model.module.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
                        else:
                            torch.save(model.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
            if results["evaluation"]["accuracy"]["overall"]+results["evaluation"]["precision_recall_f1"]["overall"][-1] >= best_accf1s[data_type]:
                best_accf1s[data_type] = results["evaluation"]["accuracy"]["overall"]+results["evaluation"]["precision_recall_f1"]["overall"][-1]
                best_accf1_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest pdtb-accf1: {:.4f} (epoch: {:0>3d})".format(
                    data_type, best_accf1s[data_type], best_accf1_epochs[data_type]))
                if args.save_best == "accf1":
                    if data_type == "valid":
                        if args.gpu_ids and len(args.gpu_ids) > 1:
                            torch.save(model.module.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
                        else:
                            torch.save(model.state_dict(), 
                                os.path.join(args.save_model_dir, "%s_best.pt" % (data_type)),
                                _use_new_zipfile_serialization=False)
    # evaluate valid_best model
    logger.info("evaluate valid_best model:")
    model.load_state_dict(torch.load(
                    os.path.join(args.save_model_dir, "valid_best.pt"),
                    map_location=device)) 
    for data_type in data_loaders:
        if data_type == 'train' or data_type == 'valid':
            continue
        
        mean_loss, results = eval_epoch(args, logger, writer,
                        model, data_type, data_loader, device, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="numer of processors")

    # data config

    parser.add_argument("--train_dataset_path", type=str,
                        help="training Dataset path")
    parser.add_argument("--valid_dataset_path", type=str,
                        help="validation Dataset path")
    parser.add_argument("--test_dataset_path", type=str,
                        help="test Dataset path")
    parser.add_argument("--blind_dataset_path", type=str, default="",
                        help="blind Dataset path")
    parser.add_argument("--pretrained_model_path", type=str, default="",
                        help="model path of pretrained NumericalModel")
    parser.add_argument("--save_model_dir", type=str,
                        help="model dir to save models")
    parser.add_argument("--num_rels", type=int, default=4,
                        choices=[4, 11, 14],
                        help="how many relations are computed")
    parser.add_argument("--min_arg", type=int, default=3,
                        help="the minimum length of arguments")
    parser.add_argument("--max_arg", type=int, default=512,
                        help="the maximum length of arguments")

    # training config
    parser.add_argument("--gpu_ids", type=str2list, default=None,
                        help="gpu ids")     
    parser.add_argument("--epochs", type=int, default=10,
                        help="epochs of training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size of training")
    parser.add_argument("--print_every", type=int, default=100,
                        help="printing log every K batchs")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="global learning rate")
    parser.add_argument("--downstream_lr", type=float, default=3e-4,
                        help="downstream learning rate")
    parser.add_argument("--save_best", type=str, default="f1",
                        choices=["loss", "acc", "f1", "accf1"],
                        help="the criteria to save best models")
    
    
    # Model config
    parser.add_argument("--encoder", type=str, default="roberta",
                        choices=["lstm", "bert", "roberta"],
                        help="the encoder")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout for neural networks")
    parser.add_argument("--use_gcn", action="store_true",
                        help="use for gcn")
    parser.add_argument("--gcn_steps", type=int, default=1,
                        help="gcn_steps")
    args = parser.parse_args()
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    args.save_model_dir = os.path.join(args.save_model_dir, ts)
    os.makedirs(args.save_model_dir, exist_ok=True)

    # save config
    save_config(args, os.path.join(args.save_model_dir, "NumericalModel.config"))

    # build logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(args.save_model_dir, "NumericalModel.log"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # build writer
    writer = SummaryWriter(args.save_model_dir)

    # train
    train(args, logger, writer)

