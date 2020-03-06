'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

def apply_masks(net, masks):
    if len(masks) > 0:
        for i in range(1, net.num_layers()):
            W = net.from_bottom_linear(i)
            W *= masks[i - 1]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--use_cnn', action="store_true")
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--prune_when_resume', action='store_true', help="compute prune matrix when resume")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

cifar10_dataset = "/checkpoint/yuandong"
save_dir = "/checkpoint/pytorch-cifar/checkpoint"

trainset = torchvision.datasets.CIFAR10(root=cifar10_dataset, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=cifar10_dataset, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# ks = [10, 15, 20, 25]

if args.use_cnn:
    ks = [64, 64, 64, 64]
    net = ModelConv((3, 32, 32), ks, 10, multi=1)
else:
    ks = [50, 75, 100, 125]
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
net = net.to(device)
masks = []
inactive_nodes = None
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.load is not None:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load)

    if isinstance(checkpoint, dict):
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        inactive_nodes = checkpoint.get("inactive_nodes", inactive_nodes)
        masks = checkpoint.get("masks", masks)
    else:
        net = checkpoint
        net = net.to(device)
        best_acc = 0
        start_epoch = 0

    if args.prune_when_resume:
        print(f"prune ratio: {ratios}")
        inactive_nodes, masks = prune(net, ratios)

        for i, (inactive, k) in enumerate(zip(inactive_nodes, ks)):
            print(f"layer{i} pruned: {len(inactive)}/{k}")

        apply_masks(net, masks)
        best_acc = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, masks, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        apply_masks(net, masks)
        inputs, targets = inputs.to(device), targets.to(device)

        if not args.use_cnn:
            inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        outputs = net(inputs)["y"]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    apply_masks(net, masks)

def test(epoch, args):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if not args.use_cnn:
                inputs = inputs.view(inputs.size(0), -1)

            outputs = net(inputs)["y"]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print(f'Saving with {acc} > {best_acc}')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            "inactive_nodes": inactive_nodes,
            "masks": masks,
            "ratios": ratios,
        }
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        filename = os.path.join(save_dir, 'ckpt.t7')
        torch.save(state, filename)
        print(f"save to {filename}")
        best_acc = acc

print("Initial test")
test(-1, args)

for epoch in range(start_epoch, start_epoch+200):
    train(epoch, masks, args)
    test(epoch, args)
