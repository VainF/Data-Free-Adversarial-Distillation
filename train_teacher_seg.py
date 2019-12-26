from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import torchvision
import network
from dataloader import get_dataloader
from utils.stream_metrics import StreamSegMetrics

from utils.visualizer import VisdomPlotter
from utils.misc import pack_images, denormalize
from collections import OrderedDict
from utils import focal_loss
import numpy as np
import random

vp = None

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        loss = focal_loss(output, target, gamma=2, ignore_index=255) #focal_loss(output, target, gamma=2)#F.cross_entropy(output, target, ignore_index=255)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    seg_metrics = StreamSegMetrics(args.num_classes)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device, dtype=torch.long)
            output = model(data)
            seg_metrics.update(output.max(1)[1].detach().cpu().numpy().astype('uint8'), target.detach().cpu().numpy().astype('uint8'))
            if i==0:
                vp.add_image( 'input', pack_images( ((data+1)/2).clamp(0, 1.0).cpu().numpy() ) )
                vp.add_image( 'target', pack_images( test_loader.dataset.decode_target(target.cpu().numpy()), channel_last=True ).astype('uint8') )
                vp.add_image( 'pred',   pack_images( test_loader.dataset.decode_target(output.max(1)[1].detach().cpu().numpy().astype('uint8')), channel_last=True ).astype('uint8') )

    results = seg_metrics.get_results()

    print('\nTest set: Acc= %.6f, mIoU: %.6f\n'%(results['Overall Acc'],results['Mean IoU']))
    return results
    
def get_model(args):
    if args.model.lower()=='deeplabv3_resnet50':
        return network.segmentation.deeplabv3.deeplabv3_resnet50(num_classes=args.num_classes, dropout_p=0.5, pretrained_backbone=True)
    elif args.model.lower()=='segnet_vgg19':
        return network.segmentation.segnet.SegNetVgg19(args.num_classes, pretrained_backbone=True)
    elif args.model.lower()=='segnet_vgg16':
        return network.segmentation.segnet.SegNetVgg16(args.num_classes, pretrained_backbone=True)
    elif args.model.lower()=='segnet_vgg13':
        return network.segmentation.segnet.SegNetVgg13(args.num_classes, pretrained_backbone=True)
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='camvid', choices=['camvid', 'nyuv2'],
                        help='dataset name (default: camvid)')
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50', choices=['deeplabv3_resnet50', 'segnet_vgg19', 'segnet_vgg16'],
                        help='model name (default: deeplabv3_resnet50)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step_size', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--scheduler', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(args)

    global vp
    vp = VisdomPlotter('15550', 'teacher-seg-%s'%args.dataset)

    train_loader, test_loader = get_dataloader(args)
    model = get_model(args)

    if args.ckpt is not None:
        model.load_state_dict( torch.load( args.ckpt ) )
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    best_result = 0
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)

    if args.test_only:
        results = test(args, model, device, test_loader)
        return

    for epoch in range(1, args.epochs + 1):
        if args.scheduler:
            scheduler.step()
        print("Lr = %.6f"%(optimizer.param_groups[0]['lr']))
        train(args, model, device, train_loader, optimizer, epoch)
        results = test(args, model, device, test_loader)
        vp.add_scalar('mIoU', epoch, results['Mean IoU'])
        if results['Mean IoU']>best_result:
            best_result = results['Mean IoU']
            torch.save(model.state_dict(),"checkpoint/teacher/%s-%s.pt"%(args.dataset, args.model))
    print("Best mIoU=%.6f"%best_result)

if __name__ == '__main__':
    main()