import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import json
import argparse
import copy
import hydra
import os
import sys

from jacobian import JacobianReg

import teacher_tune

from argparse import Namespace

import logging
log = logging.getLogger(__file__)

import basic_tools.checkpoint as checkpoint
import basic_tools.stats_op as stats
import basic_tools.utils as utils
import basic_tools.logger as logger

from model_gen import Model, ModelConv, prune
from copy import deepcopy
import pickle

from dataset import RandomDataset, init_dataset

def get_active_nodes(teacher):
    # Getting active node for teachers. 
    active_nodes = []
    for layer in range(1, teacher.num_layers()):
        W = teacher.from_bottom_linear(layer)
        if len(W.size()) == 4:
            # W: [output_filter, input_filter, x, y]
            active_nodes.append(W.permute(1, 0, 2, 3).contiguous().view(W.size(1), -1).norm(dim=1) > 1e-5)
        else:
            # W: [output_dim, input_dim]
            active_nodes.append(W.norm(dim=0) > 1e-5)

    return active_nodes

def save_model(prefix, model, i):
    filename = os.path.join(os.getcwd(), f"{prefix}-{i}.pt")
    torch.save(model, filename)
    print(f"[{i}] Saving {prefix} to {filename}")


def train_model(i, train_loader, teacher, student, train_stats_op, loss_func, perturber, optimizer, args):
    teacher.eval()
    student.train()

    train_stats_op.reset()

    if args.jacobian_reg_coeff is not None:
        reg = JacobianReg()

    def train_op(x):
        if args.jacobian_reg_coeff is not None:
            x.requires_grad = True

        optimizer.zero_grad()

        output_t = teacher(x)
        output_s = student(x)

        y_t = output_t["y"].detach()
        y_s = output_s["y"]

        if args.teacher_output_noise is not None:
            y_t = y_t + torch.randn_like(y_t) * args.teacher_output_noise

        err = loss_func(y_s, y_t)
        if torch.isnan(err).item():
            print("NAN appears, optimization aborted")
            return False

        if args.jacobian_reg_coeff is not None:
            R = reg(x, y_s)
            err = err + args.jacobian_reg_coeff * R

        err.backward()
        train_stats_op.add(output_t, output_s, y)
        optimizer.step()
        return True


    for x, y in train_loader:
        if not args.use_cnn:
            x = x.view(x.size(0), -1)
        x = x.cuda()
        y = y.cuda()

        # adversarial training, if there is anything defined. 
        if perturber is not None:
            optimizer.zero_grad()
            student.eval()
            forwarder_student = lambda x : student(x)["y"]
            forwarder_teacher = lambda x : teacher(x)["y"]
            x2 = perturber.perturb(forwarder_student, forwarder_teacher, x, loss_func) 
            student.train()

            if not train_op(x2):
                return dict(exit="nan")

        if perturber is None or args.adv_and_original:
            if not train_op(x):
                return dict(exit="nan")

        if args.normalize:
            student.normalize()

    train_stats = train_stats_op.export()

    print(f"[{i}]: Train Stats {train_stats_op.label}:")
    print(train_stats_op.prompt())

    return train_stats

def eval_model(i, eval_loader, teacher, student, eval_stats_op):
    # evaluation
    teacher.eval()
    student.eval()

    eval_stats_op.reset()

    with torch.no_grad():
        for x, y in eval_loader:
            if not teacher.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            y = y.cuda()
            output_t = teacher(x)
            output_s = student(x)

            eval_stats_op.add(output_t, output_s, y)

    eval_stats = eval_stats_op.export()

    print(f"[{i}]: Eval Stats {eval_stats_op.label}:")
    print(eval_stats_op.prompt())

    return eval_stats


def suppress_unaligned(eval_loader, teacher, student, eval_stats_op):
    # evaluation
    teacher.eval()
    student.eval()

    eval_stats_op.reset()

    with torch.no_grad():
        for x, y in eval_loader:
            if not teacher.use_cnn:
                x = x.view(x.size(0), -1)
            x = x.cuda()
            y = y.cuda()
            output_t = teacher(x)
            output_s = student(x)

            eval_stats_op.add(output_t, output_s, y)

    eval_stats = eval_stats_op.export()
    stat = eval_stats_op.find_stat_obj("StatsCorr")

    layer_idx = 0

    summary = stat.prompt()

    best_students = set(summary["best_students"][layer_idx])
    nonbest = [ i for i in range(student.ws_linear[layer_idx].weight.size(0)) if i not in best_students ]

    # Suppress others that are not best. 
    student.ws_linear[0].weight.data[nonbest,:] /= 10
    student.ws_linear[0].bias.data[nonbest] /= 10


def optimize(train_loader, eval_loader, cp, loss_func, args, lrs):
    # optimizer = optim.SGD(student.parameters(), lr = 1e-2, momentum=0.9)
    # optimizer = optim.Adam(student.parameters(), lr = 0.0001)

    if cp.epoch == 0:
        cp.stats = []
        cp.lr = lrs[0]

        print("Before optimization: ")

        if args.normalize:
            cp.student.normalize()
        
        save_model("teacher", cp.teacher, 0)

        eval_stats = eval_model(-1, eval_loader, cp.teacher, cp.student, cp.eval_stats_op)
        eval_stats["iter"] = -1
        cp.stats.append(eval_stats)

    if args.optim_method == "sgd":
        optimizer = optim.SGD(cp.student.parameters(), lr = cp.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_method == "adam":
        optimizer = optim.Adam(cp.student.parameters(), lr = cp.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unknown optim method: {args.optim_method}")

    if args.data_perturb["class"] is not None:
        perturber = hydra.utils.instantiate(args.data_perturb)
    else:
        perturber = None


    while cp.epoch < args.num_epoch:
        if cp.epoch in lrs:
            cp.lr = lrs[cp.epoch]
            print(f"[{cp.epoch}]: lr = {cp.lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = cp.lr

        train_stats = train_model(cp.epoch, train_loader, cp.teacher, cp.student, cp.train_stats_op, loss_func, perturber, optimizer, args)

        this_stats = dict(iter=cp.epoch)
        this_stats.update(train_stats)

        if "exit" in train_stats:
            cp.stats.append(this_stats)
            return cp.stats

        eval_stats = eval_model(cp.epoch, eval_loader, cp.teacher_eval, cp.student, cp.eval_stats_op)
        this_stats.update(eval_stats)

        if cp.eval_no_noise_stats_op is not None:
            this_stats.update(eval_model(cp.epoch, eval_loader, cp.teacher, cp.student, cp.eval_no_noise_stats_op))

        eval_train_stats = eval_model(cp.epoch, train_loader, cp.teacher, cp.student, cp.eval_train_stats_op)
        this_stats.update(eval_train_stats)

        print(f"[{cp.epoch}]: Bytesize of stats: {utils.count_size(this_stats) / 2 ** 20} MB")

        cp.stats.append(this_stats)

        # save student
        if args.save_student and (cp.epoch == args.num_epoch - 1 or cp.epoch % args.num_epoch_save_student == 0):
            save_model("student", cp.student, cp.epoch)

        if args.cheat_suppress_unaligned:
            print("Suppress unaligned student node.")
            suppress_unaligned(eval_loader, cp.teacher, cp.student, cp.eval_stats_op)

        print("")
        print("")

        if args.regen_dataset_each_epoch:
            train_loader.dataset.regenerate()

        cp.epoch += 1
        checkpoint.save_checkpoint(cp)

    if args.num_epoch_save_summary > 0:
        interval_stats = cp.stats[0:-1:args.num_epoch_save_summary] + [ cp.stats[-1] ]
        torch.save(interval_stats, f"summary.pth")
    else:
        end_stats = [ cp.stats[0], cp.stats[-1] ]
        torch.save(end_stats, f"summary.pth")


def parse_ks(ks_str):
    if isinstance(ks_str, str):
        if ks_str.startswith("["):
            # [20, 30, 50, 60]
            ks = eval(ks_str)
        else:
            try:
                return [ int(k) for k in ks_str.split("-") ]
            except:
                raise RuntimeError("Invalid ks: " + ks_str)
    elif isinstance(ks_str, int):
        return [ks_str]
    else:
        return ks_str


def parse_lr(lr_str):
    if isinstance(lr_str, float):
        return { 0 : lr_str }

    if lr_str.startswith("{"):
        lrs = eval(lr_str)
    else:
        items = lr_str.split("-")
        lrs = {}
        if len(items) == 1:
            # Fixed learning rate.
            lrs[0] = float(items[0])
        else:
            for k, v in zip(items[::2], items[1::2]):
                lrs[int(k)] = float(v)

    return lrs

def initialize_networks(d, ks, d_output, args):
    if not args.use_cnn:
        teacher = Model(d[0], ks, d_output, 
                has_bias=not args.no_bias, has_bn=args.teacher_bn, bn_before_relu=args.bn_before_relu, leaky_relu=args.leaky_relu).cuda()

    else:
        teacher = ModelConv(d, ks, d_output, has_bn=args.teacher_bn, bn_before_relu=args.bn_before_relu, leaky_relu=args.leaky_relu).cuda()

    if args.load_teacher is not None:
        print("Loading teacher from: " + args.load_teacher)
        checkpoint = torch.load(args.load_teacher)
        active_nodes = None
        active_ks = ks

        if isinstance(checkpoint, dict):
            teacher.load_state_dict(checkpoint['net'])

            if "inactive_nodes" in checkpoint: 
                inactive_nodes = checkpoint["inactive_nodes"]
                masks = checkpoint["masks"]
                ratios = checkpoint["ratios"]
                inactive_nodes2, masks2 = prune(teacher, ratios)

                for m, m2 in zip(masks, masks2):
                    if (m - m2).norm() > 1e-3:
                        print(m)
                        print(m2)
                        raise RuntimeError("New mask is not the same as old mask")

                for inactive, inactive2 in zip(inactive_nodes, inactive_nodes2):
                    if set(inactive) != set(inactive2):
                        raise RuntimeError("New inactive set is not the same as old inactive set")

                # Make sure the last layer is normalized. 
                # teacher.normalize_last()
                # teacher.final_w.weight.data /= 3
                # teacher.final_w.bias.data /= 3
                active_nodes = [ [ kk for kk in range(k) if kk not in a ] for a, k in zip(inactive_nodes, ks) ]
                active_ks = [ len(a) for a in active_nodes ]
        else:
            teacher = checkpoint
        
    else:
        print("Init teacher..")
        teacher.init_w(use_sep = not args.no_sep, weight_choices=list(args.weight_choices))
        if args.teacher_strength_decay > 0: 
            # Prioritize teacher node.
            print(f"Prioritize teacher node with decay coefficient: {args.teacher_strength_decay}")
            teacher.prioritize(args.teacher_strength_decay)

        if args.eval_teacher_prune_ratio > 0:
            # Prioritize teacher prune ratio
            print(f"Prune teacher node with step coefficient: {args.eval_teacher_prune_ratio}")
            teacher.prioritize_step(args.eval_teacher_prune_ratio)

        teacher.normalize()
        print("Teacher weights initiailzed randomly...")
        active_nodes = None
        active_ks = ks

    print(f"Active ks: {active_ks}")

    if args.load_student is None:
        if not args.use_cnn:
            student = Model(d[0], active_ks, d_output, 
                            multi=args.node_multi, 
                            has_bias=not args.no_bias, has_bn=args.bn, bn_before_relu=args.bn_before_relu, dropout=args.dropout).cuda()
        else:
            student = ModelConv(d, active_ks, d_output, multi=args.node_multi, has_bn=args.bn, bn_before_relu=args.bn_before_relu).cuda()

        # student can start with smaller norm. 
        student.scale(args.student_scale_down)
        initialize_student(student, teacher, args)
    else:
        print(f"Loading student {args.load_student}")
        student = torch.load(args.load_student)

    # Specify some teacher structure.
    '''
    teacher.w0.weight.data.zero_()
    span = d // ks[0]
    for i in range(ks[0]):
        teacher.w0.weight.data[i, span*i:span*i+span] = 1
    '''

    return teacher, student, active_nodes


def tune_teacher_model(teacher, train_loader, eval_loader, args):
    tune_switcher = dict(
        eval=eval_loader, 
        train=train_loader
    )

    assert args.tune_data in tune_switcher
    tune_data_loader = tune_switcher[args.tune_data]

    print(f"Tune with {args.tune_data}")

    if args.teacher_bias_tune:
        teacher_tune.tune_teacher(tune_data_loader, teacher)
    if args.teacher_bias_last_layer_tune:
        teacher_tune.tune_teacher_last_layer(tune_data_loader, teacher)

    teacher_tune.check(tune_data_loader, teacher, output_func=print)


def initialize_stats_ops_common(teacher, student, active_nodes, args):
    stats_op = stats.StatsCollector(teacher, student)

    # Compute Correlation between teacher and student activations. 
    stats_op.add_stat(stats.StatsCorr, active_nodes=active_nodes, cnt_thres=0.9)

    if args.cross_entropy:
        stats_op.add_stat(stats.StatsCELoss)
    else:
        stats_op.add_stat(stats.StatsL2Loss)

    if args.dataset in ["cifar10", "mnist"]:
        stats_op.add_stat(stats.StatsAcc)

    return stats_op


def initialize_train_stats_ops(teacher, student, active_nodes, args):
    stats_op = initialize_stats_ops_common(teacher, student, active_nodes, args)

    stats_op.add_stat(stats.StatsGrad)
    stats_op.add_stat(stats.StatsMemory)

    return stats_op


def initialize_eval_stats_ops(teacher, student, active_nodes, args):
    stats_op = initialize_stats_ops_common(teacher, student, active_nodes, args)

    if args.stats_H:
        stats_op.add_stat(stats.StatsHs)

    stats_op.add_stat(stats.WeightCorr)

    return stats_op


def initialize_loss_func(args):
    if args.cross_entropy:
        loss = nn.CrossEntropyLoss().cuda()
        def loss_func(predicted, target):
            _, target_y = target.max(1)
            return loss(predicted, target_y)
    else:
        loss_func = nn.MSELoss().cuda()

    return loss_func


def initialize_student(student, teacher, args):
    student.reset_parameters()
    # student = copy.deepcopy(student_clone)
    # student.set_teacher_sign(teacher, scale=1)
    if args.perturb is not None:
        student.set_teacher(teacher, args.perturb)
    if args.same_dir:
        student.set_teacher_dir(teacher)
    if args.same_sign:
        student.set_teacher_sign(teacher)


@hydra.main(config_path='conf/config_multilayer.yaml', strict=True)
def main(args):
    checkpoint.init_checkpoint()

    sys.stdout = logger.Logger("./log.log", mode="w") 
    sys.stderr = logger.Logger("./log.err", mode="w") 

    cmd_line = " ".join(sys.argv)
    print(f"{cmd_line}")
    print(f"Working dir: {os.getcwd()}")
    utils.set_all_seeds(args.seed)

    ks = parse_ks(args.ks)
    lrs = parse_lr(args.lr)

    if args.perturb is not None or args.same_dir or args.same_sign:
        args.node_multi = 1

    if args.load_student is not None:
        args.num_trial = 1

    if args.load_dataset_path is not None:
        train_dataset = torch.load(os.path.join(args.load_dataset_path, "train_dataset.pth"))
        eval_dataset = torch.load(os.path.join(args.load_dataset_path, "eval_dataset.pth"))
        saved = torch.load(os.path.join(args.load_dataset_path, "params_dataset.pth"))
        d = saved["d"]
        d_output = saved["d_output"]
    else:
        d, d_output, train_dataset, eval_dataset = init_dataset(args)

    if args.save_dataset:
        print("Saving training dataset")
        torch.save(train_dataset, "train_dataset.pth")
        print("Saving eval dataset")
        torch.save(eval_dataset, "eval_dataset.pth")
        print("Saving dataset params")
        torch.save(dict(d=d, d_output=d_output), "params_dataset.pth")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batchsize, shuffle=True, num_workers=4)

    if args.total_bp_iters > 0 and isinstance(train_dataset, RandomDataset):
        args.num_epoch = args.total_bp_iters / args.random_dataset_size
        if args.num_epoch != int(args.num_epoch):
            raise RuntimeError(f"random_dataset_size [{args.random_dataset_size}] cannot devide total_bp_iters [{args.total_bp_iters}]")

        args.num_epoch = int(args.num_epoch)
        print(f"#Epoch is now set to {args.num_epoch}")

    print(args.pretty())
    print(f"ks: {ks}")
    print(f"lr: {lrs}")

    if args.d_output > 0:
        d_output = args.d_output 

    print(f"d_output: {d_output}") 
    loss_func = initialize_loss_func(args)

    if args.save_train_dataset:
        print("Save training dataset")
        torch.save(train_loader, "train_dataset.pth")

    if args.save_eval_dataset:
        print("Save eval dataset")
        torch.save(eval_loader, "eval_dataset.pth")

    if checkpoint.exist_checkpoint():
        cp = checkpoint.load_checkpoint()
    elif args.resume_from_checkpoint is not None:
        cp = checkpoint.load_checkpoint(filename=args.resume_from_checkpoint)
    else:
        teacher, student, active_nodes = initialize_networks(d, ks, d_output, args)

        if args.load_teacher is None:
            tune_teacher_model(teacher, train_loader, eval_loader, args)

        if args.eval_teacher_prune_ratio > 0:
            print(f"Prune teacher weight during evaluation. Ratio: {args.eval_teacher_prune_ratio}")
            noise_teacher = deepcopy(teacher)
            noise_teacher.prune_weight_bias(args.eval_teacher_prune_ratio)
        else:
            noise_teacher = teacher
            
        print("=== Start ===")
        train_stats_op = initialize_train_stats_ops(teacher, student, active_nodes, args)
        train_stats_op.label = "train"

        eval_stats_op = initialize_eval_stats_ops(teacher, student, active_nodes, args)
        eval_stats_op.label = "eval"

        eval_train_stats_op = initialize_eval_stats_ops(teacher, student, active_nodes, args)
        eval_train_stats_op.label = "eval_train"

        if noise_teacher != teacher:
            eval_no_noise_stats_op = initialize_eval_stats_ops(teacher, student, active_nodes, args)
            eval_no_noise_stats_op.label = "eval_no_noise"
        else:
            eval_no_noise_stats_op = None

        cp = Namespace(trial_idx=0, all_stats=[], lr=None, epoch=0, \
                student=student, teacher=teacher, teacher_eval=noise_teacher, \
                train_stats_op=train_stats_op, eval_stats_op=eval_stats_op, \
                eval_train_stats_op=eval_train_stats_op, \
                eval_no_noise_stats_op=eval_no_noise_stats_op)

    # teacher.w0.bias.data.uniform_(-1, 0)
    # teacher.init_orth()

    # init_w(teacher.w0)
    # init_w(teacher.w1)
    # init_w(teacher.w2)

    # init_w2(teacher.w0, multiplier=args.init_multi)
    # init_w2(teacher.w1, multiplier=args.init_multi)
    # init_w2(teacher.w2, multiplier=args.init_multi)

    # pickle.dump(model2numpy(teacher), open("weights_gt.pickle", "wb"), protocol=2)

    while cp.trial_idx < args.num_trial:
        print(f"=== Trial {cp.trial_idx}, std = {args.data_std}, dataset = {args.dataset} ===")

        # init_corrs[-1] = predict_last_order(student, teacher, args)
        # alter_last_layer = predict_last_order(student, teacher, args)

        # import pdb
        # pdb.set_trace()
        optimize(train_loader, eval_loader, cp, loss_func, args, lrs)
        cp.all_stats.append(cp.stats)
        cp.epoch = 0
        cp.trial_idx += 1

    torch.save(cp.all_stats, "stats.pickle")

    # print("Student network")
    # print(student.w1.weight)
    # print("Teacher network")
    # print(teacher.w1.weight)
    print(f"Working dir: {os.getcwd()}")

if __name__ == "__main__":
    main()

