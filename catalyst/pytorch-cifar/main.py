'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import sys
import os

sys.path.append("../")
import attack

import os
import argparse

from models import *
from utils import progress_bar

import hydra
import basic_tools.logger as logger

def apply_masks(net, m):
    masks = m["masks"]
    if len(masks) > 0:
        for i in range(1, net.num_layers()):
            W = net.from_bottom_linear(i)
            W *= masks[i - 1]


# Training
def train(net, m, loader, optimizer, args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    attacker = attack.FastGradientSignUntargeted(
            args.epsilon, args.alpha, 
            min_val=None, max_val=None, 
            max_iters=args.k, _type=args.perturbation_type)

    for batch_idx, (inputs, targets) in enumerate(loader):
        apply_masks(net, m)
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        if not args.use_cnn:
            inputs = inputs.view(inputs.size(0), -1)

        if args.adv_training:
            net_func = lambda x: net(x)["y"]
            inputs = attacker.perturb_ce(net_func, inputs, targets, 'mean', True)

        optimizer.zero_grad()
        outputs = net(inputs)["y"]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total), stdout=sys.__stdout__)

    apply_masks(net, m)

def test(net, m, loader, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            if not args.use_cnn:
                inputs = inputs.view(inputs.size(0), -1)

            outputs = net(inputs)["y"]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), stdout=sys.__stdout__)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > m["best_acc"]:
        print(f'Saving with {acc} > {m["best_acc"]}')

        m["best_acc"] = acc
        m["net"] = net.state_dict()

        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        filename = os.path.join(args.save_dir, 'ckpt.t7')
        torch.save(m, filename)
        print(f"save to {filename}")


@hydra.main(config_path='conf/config.yaml', strict=True)
def main(args):
    sys.stdout = logger.Logger("./log.log", mode="w") 
    sys.stderr = logger.Logger("./log.err", mode="w") 

    print(os.getcwd())

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # ks = [10, 15, 20, 25]

    if args.use_cnn:
        ks = [64, 64, 64, 64]
        print(f"Use CNN, ks = {ks}")
        net = ModelConv((3, 32, 32), ks, 10, multi=1)
    else:
        ks = [50, 75, 100, 125]
        print(f"Do not use CNN, ks = {ks}")
        net = Model(3 * 32 * 32, ks, 10, has_bias=True, multi=1)

    ratios=[0.3, 0.5, 0.5, 0.7]
    # ratios=[0.3, 0.3, 0.3, 0.3]

    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    net = net.to(args.device)

    m = dict(net=net.state_dict(), best_acc=0, epoch=0, masks=[], inactive_nodes=None, ratios=ratios)

    if args.device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.load is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.load)

        if "net" not in checkpoint:
            net.load_state_dict(checkpoint)
            m = dict(net=m, best_acc=0, epoch=0, masks=[], inactive_nodes=None, ratios=ratios)
        else:
            net.load_state_dict(checkpoint["net"])
            m.update(checkpoint)

        if args.prune_when_resume:
            print(f"prune ratio: {ratios}")
            inactive_nodes, masks = prune(m["net"], ratios)
            for i, (inactive, k) in enumerate(zip(inactive_nodes, ks)):
                print(f"layer{i} pruned: {len(inactive)}/{k}")

            m["inactive_nodes"] = inactive_nodes
            m["masks"] = masks

            apply_masks(net, m)
            m["best_acc"] = 0

    if args.method == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.method == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    print("Initial test")
    test(net, m, testloader, args)

    start_epoch = m["epoch"]
    while m["epoch"] < start_epoch + 200:
        print(f'\nEpoch: {m["epoch"]}')
        train(net, m, trainloader, optimizer, args)
        test(net, m, testloader, args)
        m["epoch"] += 1


if __name__ == "__main__":
    main()
