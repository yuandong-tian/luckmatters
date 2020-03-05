import sys
import os
import torch
import argparse
import torch.nn as nn

from copy import deepcopy

from model_gen import *

def convert(s):
    if isinstance(s, str):
        return [ int(v) for v in s.split("-") ]
    else:
        return s


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--d', type=str, default="3-32-32")
    parser.add_argument('--d_output', type=int, default=10)
    parser.add_argument('--ks', type=str, default="64-64-64-64")
    parser.add_argument('--bn', action="store_true")
    parser.add_argument('--bn_before_relu', action="store_true")
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    d = convert(args.d)
    ks = convert(args.ks)
    d_output = convert(args.d_output)

    model = ModelConv(d, ks, d_output, has_bn=args.bn, bn_before_relu=args.bn_before_relu, leaky_relu=False).cuda()

    checkpoint = torch.load(args.model)
    net = checkpoint["net"]
    model.load_state_dict(net)
    model.cuda()

    old_model = deepcopy(model)

    assert "inactive_nodes" in checkpoint
    inactive_nodes = checkpoint["inactive_nodes"]
    # Then we kill inactive nodes  

    input_channel, height, width = d
    last_active_node = list(range(input_channel))
    depth = len(model.ws_linear)

    ws_linear = nn.ModuleList()
    ws_bn = nn.ModuleList()
    active_nodes = []

    # a fake input to test whether there is any error.
    batchsize = 4
    x = torch.FloatTensor(batchsize, *d).normal_(0, 10).cuda()

    for d in range(depth):
        w = model.ws_linear[d]
        num_nodes = w.weight.size(0)

        inactive_node = inactive_nodes[d]
        active_node = [ i for i in range(num_nodes) if i not in inactive_node ]
        active_nodes.append(active_node)

        print(f"#active: {len(active_node)}/{num_nodes} ({len(active_node)/num_nodes})")
        # active_node = active_nodes[d]

        new_w = nn.Conv2d(len(last_active_node), len(active_node), w.kernel_size, padding=w.padding)

        # import pdb
        # pdb.set_trace()
        if any(map(lambda x: x < 0 or x >= num_nodes, active_node)):
            print(f"Active node OOB: {num_nodes}")
            print(active_node)

        if any(map(lambda x: x < 0 or x >= w.weight.size(1), last_active_node)):
            print(f"Last active node OOB: {w.weight.size(1)}")
            print(last_active_node)

        new_w.weight.data[:] = w.weight.data[active_node, :, :, :][:, last_active_node, :, :]
        new_w.bias.data[:] = w.bias.data[active_node]

        ws_linear.append(new_w)

        #
        if args.bn:
            w_bn = model.ws_bn[d]
            state = w_bn.state_dict()
            state2 = { k : v[active_node] if v.dim() > 0 else v for k, v in state.items() }

            new_w_bn = nn.BatchNorm2d(len(active_node))
            new_w_bn.load_state_dict(state2)

            ws_bn.append(new_w_bn)

        last_active_node = active_node
        last_num_nodes = num_nodes
        height -= 2
        width -= 2

    model.ws_linear = ws_linear
    model.ws_bn = ws_bn

    num_nodes = model.final_w.weight.size(1)

    assert height * width * last_num_nodes == num_nodes 

    # The final linear layer. 
    active_map = torch.BoolTensor(last_num_nodes, height * width).fill_(0)
    for k in last_active_node:
        active_map[k, :] = True

    active_node = active_map.view(-1).nonzero().squeeze().tolist()

    print(f"height = {height}, width = {width}")
    print(f"final_w #input node active: {len(active_node)}/{num_nodes} ({len(active_node)/num_nodes})")

    new_w = nn.Linear(len(active_node), d_output)
    new_w.weight.data[:] = model.final_w.weight.data[:, active_node]
    new_w.bias.data[:] = model.final_w.bias.data

    model.final_w = new_w
    model.cuda()

    model.eval()
    old_model.eval()

    # compare two models.
    with torch.no_grad():
        output_new = model(x)
        output_old = old_model(x)

    for d in range(depth):
        active_node = active_nodes[d]

        diff = output_old["hs"][d][:, active_node, :, :] - output_new["hs"][d]
        norm = diff.norm() 
        print(f"L{d}: diff norm = {norm}")
        if norm > 1e-4:
            import pdb
            pdb.set_trace()

    diff = output_old["y"] - output_new["y"]
    norm = diff.norm() 
    print(f"Final layer: diff norm = {norm}")

    print(f"Saving model to {args.output}")
    torch.save(model, args.output)

if __name__ == "__main__":
    main()

