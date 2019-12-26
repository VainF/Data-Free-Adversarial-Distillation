from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import network
from dataloader import get_dataloader

import random
import numpy as np
import os

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, cur_epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        cur_epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct/len(test_loader.dataset)
    
def get_model(args):
    if args.model.lower()=='lenet5':
        return network.lenet.LeNet5()
    elif args.model.lower()=='resnet34':
        return  torchvision.models.resnet34(num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.model.lower()=='resnet34_8x':
        return network.resnet.resnet34(num_classes=args.num_classes)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'svhn', 'cifar10', 'caltech101', 'nyuv2'],
                        help='dataset name (default: mnist)')
    parser.add_argument('--model', type=str, default='lenet5', choices=['lenet5', 'resnet34', 'resnet34_8x'],
                        help='model name (default: mnist)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step_size', type=int, default=50, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs('checkpoint/teacher', exist_ok=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)

    train_loader, test_loader = get_dataloader(args)
    model = get_model(args)

    if args.ckpt is not None:
        model.load_state_dict( torch.load( args.ckpt ) )
    
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    best_acc = 0
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)

    if args.test_only:
        acc = test(args, model, device, test_loader, 0)
        return
    
    for epoch in range(1, args.epochs + 1):
        if args.scheduler:
            scheduler.step()
        #print("Lr = %.6f"%(optimizer.param_groups[0]['lr']))
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader, epoch)
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(),"checkpoint/teacher/%s-%s.pt"%(args.dataset, args.model))
    print("Best Acc=%.6f"%best_acc)

if __name__ == '__main__':
    main()