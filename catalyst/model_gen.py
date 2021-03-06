import sys
import os

import torch
import torch.nn as nn
import random
import basic_tools.theory as theory

# Generate random orth matrix.
import numpy as np
import math

def get_aug_w(w):
    # w: [output_d, input_d]
    # aug_w: [output_d + 1, input_d + 1]
    output_d, input_d = w.weight.size()
    aug_w = torch.zeros( (output_d + 1, input_d + 1), dtype = w.weight.dtype, device = w.weight.device)
    aug_w[:output_d, :input_d] = w.weight.data
    aug_w[:output_d, input_d] = w.bias.data
    aug_w[output_d, input_d] = 1
    return aug_w

def set_orth(layer):
    w = layer.weight
    orth = theory.haar_measure(w.size(1))
    w.data = torch.from_numpy(orth[:w.size(0), :w.size(1)].astype('f4')).cuda()

def set_add_noise(layer, teacher_layer, perturb):
    layer.weight.data[:] = teacher_layer.weight.data[:] + torch.randn(teacher_layer.weight.size()).cuda() * perturb
    layer.bias.data[:] = teacher_layer.bias.data[:] + torch.randn(teacher_layer.bias.size()).cuda() * perturb

def set_same_dir(layer, teacher_layer):
    norm = layer.weight.data.norm()
    r = norm / teacher_layer.weight.data.norm()
    layer.weight.data[:] = teacher_layer.weight.data * r
    layer.bias.data[:] = teacher_layer.bias.data * r

def set_same_sign(layer, teacher_layer):
    sel = (teacher_layer.weight.data > 0) * (layer.weight.data < 0) + (teacher_layer.weight.data < 0) * (layer.weight.data > 0)
    layer.weight.data[sel] *= -1.0

    sel = (teacher_layer.bias.data > 0) * (layer.bias.data < 0) + (teacher_layer.bias.data < 0) * (layer.bias.data > 0)
    layer.bias.data[sel] *= -1.0

def normalize_layer(layer):
    # [output, input]
    w = layer.weight.data
    for i in range(w.size(0)):
        norm = w[i].pow(2).sum().sqrt() + 1e-5 
        w[i] /= norm
        if layer.bias is not None:
            layer.bias.data[i] /= norm

def init_w(layer, use_sep=True, weight_choices=[-0.5, -0.25, 0, 0.25, 0.5]):
    sz = layer.weight.size()
    output_d = sz[0]
    input_d = 1
    for s in sz[1:]:
        input_d *= s

    if use_sep:
        layer.weight.data[:] = torch.from_numpy(theory.init_separate_w(output_d, input_d, weight_choices)).view(*sz).cuda()
        if layer.bias is not None:
            layer.bias.data.uniform_(-.5, 0.5)

def init_w2(w, multiplier=5):
    w.weight.data *= multiplier
    w.bias.data.normal_(0, std=1)
    # w.bias.data *= 5
    for i, ww in enumerate(w.weight.data):
        pos_ratio = (ww > 0.0).sum().item() / w.weight.size(1) - 0.5
        w.bias.data[i] -= pos_ratio


def prune_weight_bias(w, prune_ratio):
    w.weight.data += torch.zeros_like(w.weight.data).normal_(0, std=noise_level)
    w.bias.data += torch.zeros_like(w.bias.data).normal_(0, std=noise_level)


class Model(nn.Module):
    def __init__(self, d, ks, d_output, multi=1, has_bn=True, has_bias=True, bn_before_relu=False, leaky_relu=None, dropout=0.0):
        super(Model, self).__init__()
        self.d = d
        self.ks = ks
        self.has_bn = has_bn
        self.ws_linear = nn.ModuleList()
        self.ws_bn = nn.ModuleList()
        self.bn_before_relu = bn_before_relu
        last_k = d
        self.sizes = [d]

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        for k in ks:
            k = int(k * multi + 0.5)
            self.ws_linear.append(nn.Linear(last_k, k, bias=has_bias))
            if has_bn:
                self.ws_bn.append(nn.BatchNorm1d(k, affine=True))
            self.sizes.append(k)
            last_k = k

        self.final_w = nn.Linear(last_k, d_output, bias=has_bias)
        self.relu = nn.ReLU() if leaky_relu is None else nn.LeakyReLU(leaky_relu)

        self.sizes.append(d_output)

        self.use_cnn = False

    def init_orth(self):
        for w in self.ws:
            set_orth(w)
        set_orth(self.final_w)

    def set_teacher(self, teacher, perturb):
        for w_s, w_t in zip(self.ws, teacher.ws):
            set_add_noise(w_s, w_t, perturb)
        set_add_noise(self.final_w, teacher.final_w, perturb)

    def set_teacher_dir(self, teacher):
        for w_s, w_t in zip(self.ws, teacher.ws):
            set_same_dir(w_s, w_t)
        set_same_dir(self.final_w, teacher.final_w)

    def set_teacher_sign(self, teacher):
        for w_s, w_t in zip(self.ws, teacher.ws):
            set_same_sign(w_s, w_t)
        set_same_sign(self.final_w, teacher.final_w)

    def prioritize(self, strength_decay):
        def _prioritize(w):
            # output x input.
            '''
            for i in range(w.weight.size(0)):
                c = pow(1 + i, strength_decay)
                w.weight.data[i, :] /=  c
                w.bias.data[i] /= c
            '''
            for i in range(w.weight.size(1)):
                w.weight.data[:, i] /= pow(1 + i, strength_decay) 
            
        # Prioritize teacher node.
        for w in self.ws_linear[1:]:
            _prioritize(w)

        _prioritize(self.final_w)

    def prioritize_step(self, prune_ratio):
        def _prioritize(w):
            # output x input.
            '''
            for i in range(w.weight.size(0)):
                c = pow(1 + i, strength_decay)
                w.weight.data[i, :] /=  c
                w.bias.data[i] /= c
            '''
            n_keep = int(w.weight.size(1) * (1 - prune_ratio))
            # Make them insignificant
            w.weight.data[:, n_keep:] /= 3
            
        # Prioritize teacher node.
        for w in self.ws_linear[1:]:
            _prioritize(w)

        _prioritize(self.final_w)

    def scale(self, r):
        def _scale(w):
            w.weight.data *= r
            w.bias.data *= r

        for w in self.ws_linear:
            _scale(w)

        _scale(self.final_w)

    def prune_weight_bias(self, prune_ratio):
        def _prune(w):
            # output x input.
            n_keep = int(w.weight.size(0) * (1 - prune_ratio))
            w.weight.data[n_keep:, :] = 0
            w.bias.data[n_keep:] = 0
            
        # Prioritize teacher node.
        for w in self.ws_linear:
            _prune(w)

        # _prune(self.final_w)

    def forward(self, x):
        hs = []
        pre_bns = []
        post_lins = []
        #bns = []
        h = x
        for i in range(len(self.ws_linear)):
            w = self.ws_linear[i]
            h = w(h)
            post_lins.append(h)
            if self.bn_before_relu:
                pre_bns.append(h)
                if len(self.ws_bn) > 0:
                    bn = self.ws_bn[i]
                    h = bn(h)
                h = self.relu(h)
            else:
                h = self.relu(h)
                pre_bns.append(h)
                if len(self.ws_bn) > 0:
                    bn = self.ws_bn[i]
                    h = bn(h)

            if self.dropout is not None:
                h = self.dropout(h)

            hs.append(h)
            #bns.append(h)

        y = self.final_w(hs[-1])
        return dict(hs=hs, post_lins=post_lins, pre_bns=pre_bns, y=y)

    def init_w(self, use_sep=True, weight_choices=None):
        for w in self.ws_linear:
            init_w(w, use_sep=use_sep, weight_choices=weight_choices)
        init_w(self.final_w, use_sep=use_sep, weight_choices=weight_choices)

    def reset_parameters(self):
        for w in self.ws_linear:
            w.reset_parameters()
        for w in self.ws_bn:
            w.reset_parameters()
        self.final_w.reset_parameters()

    def normalize(self):
        for w in self.ws_linear:
            normalize_layer(w)
        normalize_layer(self.final_w)

    def from_bottom_linear(self, j):
        if j < len(self.ws_linear):
            return self.ws_linear[j].weight.data
        elif j == len(self.ws_linear):
            return self.final_w.weight.data
        else:
            raise RuntimeError("j[%d] is out of bound! should be [0, %d]" % (j, len(self.ws)))

    def from_bottom_aug_w(self, j):
        if j < len(self.ws_linear):
            return get_aug_w(self.ws_linear[j])
        elif j == len(self.ws_linear):
            return get_aug_w(self.final_w)
        else:
            raise RuntimeError("j[%d] is out of bound! should be [0, %d]" % (j, len(self.ws)))

    def num_hidden_layers(self):
        return len(self.ws_linear)

    def num_layers(self):
        return len(self.ws_linear) + 1

    def from_bottom_bn(self, j):
        assert j < len(self.ws_bn)
        return self.ws_bn[j]


class ModelConv(nn.Module):
    def __init__(self, input_size, ks, d_output, multi=1, has_bn=True, bn_before_relu=False, leaky_relu=None):
        super(ModelConv, self).__init__()
        self.ks = ks
        self.ws_linear = nn.ModuleList()
        self.ws_bn = nn.ModuleList()
        self.bn_before_relu = bn_before_relu

        init_k, h, w = input_size
        last_k = init_k

        for k in ks:
            k = int(k * multi + 0.5)
            self.ws_linear.append(nn.Conv2d(last_k, k, 3))
            if has_bn:
                self.ws_bn.append(nn.BatchNorm2d(k))
            last_k = k
            h -= 2
            w -= 2

        self.final_w = nn.Linear(last_k * h * w, d_output)
        self.relu = nn.ReLU() if leaky_relu is None else nn.LeakyReLU(leaky_relu)

        self.use_cnn = True

    def scale(self, r):
        def _scale(w):
            w.weight.data *= r
            w.bias.data *= r

        for w in self.ws_linear:
            _scale(w)

        _scale(self.final_w)

    def forward(self, x):
        hs = []
        post_lins = []
        #bns = []
        h = x
        for i in range(len(self.ws_linear)):
            w = self.ws_linear[i]
            h = w(h)
            post_lins.append(h)
            if self.bn_before_relu:
                if len(self.ws_bn) > 0:
                    bn = self.ws_bn[i]
                    h = bn(h)
                h = self.relu(h)
            else:
                h = self.relu(h)
                if len(self.ws_bn) > 0:
                    bn = self.ws_bn[i]
                    h = bn(h)
            hs.append(h)
            #bns.append(h)
        h = hs[-1].view(h.size(0), -1)
        y = self.final_w(h)
        return dict(hs=hs, post_lins=post_lins, y=y)

    def init_w(self, use_sep=True, weight_choices=None):
        for w in self.ws_linear:
            init_w(w, use_sep=use_sep, weight_choices=weight_choices)
        init_w(self.final_w, use_sep=use_sep, weight_choices=weight_choices)

    def normalize(self):
        for w in self.ws_linear:
            normalize_layer(w)
        normalize_layer(self.final_w)

    def normalize_last(self):
        normalize_layer(self.final_w)

    def reset_parameters(self):
        for w in self.ws_linear:
            w.reset_parameters()
        for w in self.ws_bn:
            w.reset_parameters()
        self.final_w.reset_parameters()

    def from_bottom_linear(self, j):
        if j < len(self.ws_linear):
            return self.ws_linear[j].weight.data
        elif j == len(self.ws_linear):
            return self.final_w.weight.data
        else:
            raise RuntimeError("j[%d] is out of bound! should be [0, %d]" % (j, len(self.ws)))

    def num_hidden_layers(self):
        return len(self.ws_linear)

    def num_layers(self):
        return len(self.ws_linear) + 1

    def from_bottom_bn(self, j):
        assert j < len(self.ws_bn)
        return self.ws_bn[j]


def prune(net, ratios):
    # Prune the network and finetune. 
    n = net.num_layers()
    # Compute L1 norm and and prune them globally
    masks = []
    inactive_nodes = []
    for i in range(1, n):
        W = net.from_bottom_linear(i)
        # Prune all input neurons
        input_dim = W.size(1)
        fc_to_conv = False

        if isinstance(net, ModelConv):
            if len(W.size()) == 4:
                # W: [output_filter, input_filter, x, y]
                w_norms = W.permute(1, 0, 2, 3).contiguous().view(W.size(1), -1).abs().mean(1)
            else:
                # The final FC layer. 
                input_dim = net.from_bottom_linear(i - 1).size(0)
                W_reshaped = W.view(W.size(0), input_dim, -1).permute(0, 2, 1)
                w_norms = W_reshaped.reshape(-1, input_dim).abs().mean(0)
                fc_to_conv = True
        else:
            # W: [output_dim, input_dim]
            w_norms = W.abs().mean(0)

        sorted_w, sorted_indices = w_norms.sort(0)
        n_pruned = int(input_dim * ratios[i - 1])
        inactive_mask = sorted_indices[:n_pruned]

        m = W.clone().fill_(1.0)
        if fc_to_conv:
            m = m.view(m.size(0), input_dim, -1) 
            m[:, inactive_mask, :] = 0
            m = m.view(W.size(0), W.size(1))
        else:
            m[:, inactive_mask] = 0

        # Set the mask for the lower layer to zero. 
        inactive_nodes.append(inactive_mask.cpu().tolist())
        masks.append(m)
        
    return inactive_nodes, masks

