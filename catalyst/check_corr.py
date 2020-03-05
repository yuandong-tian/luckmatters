import sys
import os
import torch
import argparse
import basic_tools.stats_op as stats
import torchvision as tv
from torch.utils.data import DataLoader 
import torch.nn as nn

from collections import OrderedDict

from model_gen import *
from copy import deepcopy

sys.path.append("/private/home/yuandong/forked/TRADES/models")
from wideresnet import WideResNet

def register_hook(model, cls):
    hs = []

    def relu_hook(module, input, output):
        hs.append(output)

    for k, v in model.named_modules():
        if isinstance(v, cls):
            v.register_forward_hook(relu_hook)

    return hs

def load_model(ref, filename):
    data = torch.load(filename)
    if isinstance(data, OrderedDict):
        model = deepcopy(ref)
        model.load_state_dict(data)
        return model
    else:
        return data

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='mnist', help='use what dataset')
    parser.add_argument('--use', default='eval', help='whether we use train or eval')
    parser.add_argument('--data_root', default='/checkpoint/yuandong', 
        help='the directory to save the dataset')
    parser.add_argument('--model0', type=str, default=None)
    parser.add_argument('--model1', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    # parameters for generating adversarial examples

    if args.dataset == "mnist":
        factory = tv.datasets.MNIST
        transform_train = tv.transforms.ToTensor()
        transform_eval = tv.transforms.ToTensor()
        input_channel = 1

    elif args.dataset == "cifar10":
        factory = tv.datasets.CIFAR10
        transform_train = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                    (4,4,4,4), mode='constant', value=0).squeeze()),
                tv.transforms.ToPILImage(),
                tv.transforms.RandomCrop(32),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
        transform_eval = tv.transforms.ToTensor()
        input_channel = 3
        model = WideResNet().cuda()

    else:
        raise NotImplementedError

    tr_dataset = factory(args.data_root, 
            train=True, 
            transform=transform_train,
            download=True)

    # evaluation during training
    te_dataset = factory(args.data_root, 
            train=False, 
            transform=transform_eval, 
            download=True)

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    loader = te_loader if args.use == "eval" else tr_loader

    # Compare outputs from two different models.
    model0 = load_model(model, args.model0)
    model1 = load_model(model, args.model1)

    model0.cuda()
    model1.cuda()

    model0.eval()
    model1.eval()

    hs0 = dict()
    hs1 = dict()

    hs0["hs"] = register_hook(model0, nn.Conv2d)
    hs1["hs"] = register_hook(model1, nn.Conv2d)

    num_same_labels = 0
    acc0 = 0
    acc1 = 0
    n = 0

    stats01 = stats.StatsCorr(model0, model1)
    stats10 = stats.StatsCorr(model1, model0)

    for data, label in loader:
        data, label = data.cuda(), label.cuda()

        del hs0["hs"][:]
        del hs1["hs"][:]

        with torch.no_grad():
            output0 = model0(data)
            output1 = model1(data)

        if isinstance(output0, dict):
            output0 = output0["y"]

        if isinstance(output1, dict):
            output1 = output1["y"]

        pred0 = torch.max(output0, dim=1)[1]
        pred1 = torch.max(output1, dim=1)[1]

        stats01.add(hs0, hs1, label)
        stats10.add(hs1, hs0, label)

        num_same_labels += (pred0 == pred1).sum().item()
        acc0 += (pred0 == label).sum().item()
        acc1 += (pred1 == label).sum().item()

        n += pred0.size(0)

    stats01.export()
    stats10.export()

    summary01 = stats01.prompt(verbose=True)
    summary10 = stats10.prompt(verbose=True)

    result = dict(
        summary01=summary01, 
        summary10=summary10,
        acc0=acc0/n,
        acc1=acc1/n,
        ratio_same_label=num_same_labels/n
    ) 

    torch.save(result, args.output)

    print(f"accu0: {acc0/n} ({acc0}/{n})") 
    print(f"accu1: {acc1/n} ({acc1}/{n})") 
    print(f"ratio of same label: {num_same_labels/n} ({num_same_labels}/{n})") 

if __name__ == "__main__":
    main()
